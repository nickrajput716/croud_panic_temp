import os
import json
import time
import base64
import threading
import uuid
from io import BytesIO

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, session
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'}

# In-memory job store  {job_id: {status, progress, result}}
jobs = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────
#  COMPUTER VISION CORE
# ──────────────────────────────────────────────

def compute_direction_entropy(flow):
    """
    Direction entropy of optical-flow field.
    High entropy → chaotic / random movement → panic signal.
    Returns value in [0, 1].
    """
    angles = np.arctan2(flow[..., 1], flow[..., 0])
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    return float(entropy / np.log(8))   # normalise to [0,1]


def compute_crowd_density(frame, bg_subtractor):
    """
    Foreground pixel ratio via MOG2 background subtraction.
    Returns value in [0, 1].
    """
    fg = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    return float(np.count_nonzero(fg) / fg.size)


def compute_flow_divergence(flow):
    """
    Positive divergence → people spreading outward (escape pattern).
    Returns mean absolute divergence normalised to [0, 1].
    """
    fx = flow[..., 0]
    fy = flow[..., 1]
    dfx_dx = np.gradient(fx, axis=1)
    dfy_dy = np.gradient(fy, axis=0)
    divergence = dfx_dx + dfy_dy
    return float(np.mean(np.abs(divergence)) / 10.0)   # soft-clip


def compute_turbulence(magnitude):
    """
    Spatial variance of speed — high turbulence = stampede-like mixing.
    """
    return float(np.var(magnitude))


def panic_score_from_metrics(speed, entropy, density, accel, turbulence, divergence):
    """
    Weighted fusion of six CV signals → panic score in [0, 100].

    Weights chosen to emphasise chaotic motion (entropy) and
    sudden acceleration over raw density alone.
    """
    w_speed     = 0.20
    w_entropy   = 0.30
    w_density   = 0.12
    w_accel     = 0.15
    w_turbulence= 0.13
    w_divergence= 0.10

    s_speed      = min(speed / 12.0, 1.0)
    s_entropy    = min(entropy, 1.0)
    s_density    = min(density * 4.0, 1.0)
    s_accel      = min(accel / 6.0, 1.0)
    s_turbulence = min(turbulence / 60.0, 1.0)
    s_divergence = min(divergence * 3.0, 1.0)

    raw = (w_speed * s_speed + w_entropy * s_entropy +
           w_density * s_density + w_accel * s_accel +
           w_turbulence * s_turbulence + w_divergence * s_divergence)

    return min(raw * 100.0, 100.0)


def risk_label(score):
    if score < 25:  return "NORMAL"
    if score < 45:  return "LOW"
    if score < 65:  return "MODERATE"
    if score < 80:  return "HIGH"
    return "CRITICAL"


def recommendation_text(overall_risk, peak_ts):
    msgs = {
        "NORMAL":   "Crowd behaviour is normal. Routine monitoring is sufficient.",
        "LOW":      f"Minor anomalies detected near {peak_ts}s. Increase monitoring frequency.",
        "MODERATE": f"Moderate panic indicators around {peak_ts}s. Deploy crowd-management personnel.",
        "HIGH":     f"HIGH RISK at {peak_ts}s! Initiate crowd-dispersal protocols immediately.",
        "CRITICAL": f"CRITICAL at {peak_ts}s! Activate emergency response — clear evacuation routes NOW.",
    }
    return msgs.get(overall_risk, "")


def render_motion_frame(frame, flow, panic_score):
    """
    Draw optical-flow arrows and panic overlay on frame.
    Returns JPEG bytes (base64-encoded string).
    """
    vis = frame.copy()
    h, w = vis.shape[:2]
    step = 24

    # Draw flow vectors
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            mag = np.sqrt(fx**2 + fy**2)
            if mag < 0.5:
                continue
            # Colour by speed: green→yellow→red
            ratio = min(mag / 10.0, 1.0)
            color = (0, int(255 * (1 - ratio)), int(255 * ratio))
            end = (int(x + fx * 3), int(y + fy * 3))
            cv2.arrowedLine(vis, (x, y), end, color, 1, tipLength=0.4)

    # Panic score banner
    risk = risk_label(panic_score)
    banner_colors = {
        "NORMAL": (0, 180, 0), "LOW": (0, 220, 100),
        "MODERATE": (0, 180, 255), "HIGH": (0, 80, 255), "CRITICAL": (0, 0, 220)
    }
    bcolor = banner_colors.get(risk, (128, 128, 128))
    cv2.rectangle(vis, (0, 0), (w, 38), bcolor, -1)
    cv2.putText(vis, f"Panic: {panic_score:.1f}/100  |  Risk: {risk}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode()


# ──────────────────────────────────────────────
#  VIDEO ANALYSIS WORKER
# ──────────────────────────────────────────────

def analyse_video_worker(job_id, video_path):
    job = jobs[job_id]
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            job['status'] = 'error'
            job['error'] = 'Cannot open video file.'
            return

        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration   = total_fr / fps

        bg_sub = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40)

        ret, prev_frame = cap.read()
        if not ret:
            job['status'] = 'error'
            job['error'] = 'Video is empty.'
            return

        prev_frame = cv2.resize(prev_frame, (640, 360))
        prev_gray  = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        sample_interval = max(1, int(fps / 6))  # ~6 samples/s
        frame_num = 0
        analyses  = []
        key_frames = []           # store a few annotated frames
        prev_speed = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1

            # Progress
            job['progress'] = min(int(frame_num / max(total_fr, 1) * 100), 99)

            if frame_num % sample_interval != 0:
                continue

            frame = cv2.resize(frame, (640, 360))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            flow  = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

            magnitude  = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_speed  = float(np.mean(magnitude))
            entropy    = compute_direction_entropy(flow)
            density    = compute_crowd_density(frame, bg_sub)
            turb       = compute_turbulence(magnitude)
            div        = compute_flow_divergence(flow)
            accel      = abs(avg_speed - prev_speed)
            prev_speed = avg_speed

            score = panic_score_from_metrics(avg_speed, entropy, density, accel, turb, div)
            risk  = risk_label(score)
            ts    = round(frame_num / fps, 2)

            rec = {
                'frame':     frame_num,
                'timestamp': ts,
                'panic_score': round(score, 2),
                'avg_speed':   round(avg_speed, 4),
                'entropy':     round(entropy, 4),
                'density':     round(density, 4),
                'turbulence':  round(turb, 4),
                'divergence':  round(div, 4),
                'acceleration':round(accel, 4),
                'risk_level':  risk,
            }
            analyses.append(rec)

            # Save a few key-frame thumbnails (max 12)
            if len(key_frames) < 12:
                img_b64 = render_motion_frame(frame, flow, score)
                key_frames.append({'timestamp': ts, 'score': round(score,1),
                                   'risk': risk, 'image': img_b64})

            prev_gray = gray

        cap.release()

        if not analyses:
            job['status'] = 'error'
            job['error'] = 'No frames could be processed.'
            return

        scores     = [a['panic_score'] for a in analyses]
        max_score  = max(scores)
        avg_score  = sum(scores) / len(scores)
        peak_idx   = scores.index(max_score)
        peak_ts    = analyses[peak_idx]['timestamp']
        overall    = risk_label(max_score)

        job['result'] = {
            'total_frames_analysed': len(analyses),
            'duration_seconds': round(duration, 2),
            'fps': round(fps, 1),
            'max_panic_score': round(max_score, 2),
            'avg_panic_score': round(avg_score, 2),
            'peak_panic_timestamp': round(peak_ts, 2),
            'overall_risk': overall,
            'recommendation': recommendation_text(overall, peak_ts),
            'frame_analyses': analyses,
            'key_frames': key_frames,
            'chart': {
                'timestamps':  [a['timestamp']   for a in analyses],
                'panic_scores':[a['panic_score']  for a in analyses],
                'speeds':      [a['avg_speed']    for a in analyses],
                'densities':   [a['density']      for a in analyses],
                'entropies':   [a['entropy']      for a in analyses],
            },
            'metrics_summary': {
                'max_speed':    round(max(a['avg_speed']    for a in analyses), 4),
                'max_entropy':  round(max(a['entropy']      for a in analyses), 4),
                'max_density':  round(max(a['density']      for a in analyses), 4),
                'max_accel':    round(max(a['acceleration'] for a in analyses), 4),
            }
        }
        job['progress'] = 100
        job['status']   = 'done'

    except Exception as e:
        job['status'] = 'error'
        job['error']  = str(e)


# ──────────────────────────────────────────────
#  LIVE WEBCAM FRAME PROCESSING
# ──────────────────────────────────────────────

live_sessions = {}   # session_id → {prev_gray, bg_sub, prev_speed, history}


def process_live_frame(session_id, frame_b64):
    """Process one base64-encoded JPEG frame from webcam."""
    # Decode
    img_bytes = base64.b64decode(frame_b64.split(',')[-1])
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    frame = cv2.resize(frame, (640, 360))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Init or retrieve session state
    if session_id not in live_sessions:
        live_sessions[session_id] = {
            'prev_gray':  gray,
            'bg_sub':     cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40),
            'prev_speed': 0.0,
            'history':    [],
            'last_access':time.time()
        }
        return {'status': 'initialising', 'panic_score': 0, 'risk': 'NORMAL',
                'metrics': {}, 'frame_b64': None}

    s = live_sessions[session_id]
    s['last_access'] = time.time()

    flow = cv2.calcOpticalFlowFarneback(
        s['prev_gray'], gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    magnitude  = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    avg_speed  = float(np.mean(magnitude))
    entropy    = compute_direction_entropy(flow)
    density    = compute_crowd_density(frame, s['bg_sub'])
    turb       = compute_turbulence(magnitude)
    div        = compute_flow_divergence(flow)
    accel      = abs(avg_speed - s['prev_speed'])
    s['prev_speed'] = avg_speed

    score = panic_score_from_metrics(avg_speed, entropy, density, accel, turb, div)
    risk  = risk_label(score)

    s['prev_gray'] = gray
    s['history'].append(round(score, 2))
    if len(s['history']) > 60:
        s['history'].pop(0)

    annotated_b64 = render_motion_frame(frame, flow, score)

    return {
        'status':      'ok',
        'panic_score': round(score, 2),
        'risk':        risk,
        'metrics': {
            'speed':    round(avg_speed, 3),
            'entropy':  round(entropy, 3),
            'density':  round(density, 3),
            'accel':    round(accel, 3),
        },
        'history':    s['history'],
        'frame_b64':  annotated_b64,
    }


# ──────────────────────────────────────────────
#  FLASK ROUTES
# ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return redirect(url_for('index'))
    f = request.files['video']
    if f.filename == '' or not allowed_file(f.filename):
        return redirect(url_for('index'))

    filename = secure_filename(f.filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'running', 'progress': 0, 'result': None, 'error': None}

    t = threading.Thread(target=analyse_video_worker, args=(job_id, path), daemon=True)
    t.start()

    return redirect(url_for('progress_page', job_id=job_id))


@app.route('/progress/<job_id>')
def progress_page(job_id):
    return render_template('progress.html', job_id=job_id)


@app.route('/api/job/<job_id>')
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({
        'status':   job['status'],
        'progress': job['progress'],
        'error':    job.get('error'),
    })


@app.route('/result/<job_id>')
def result_page(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'done':
        return redirect(url_for('progress_page', job_id=job_id))
    return render_template('result.html', result=job['result'], job_id=job_id)


@app.route('/api/result/<job_id>')
def get_result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Not found'}), 404
    if job['status'] != 'done':
        return jsonify({'status': job['status']}), 202
    return jsonify(job['result'])


@app.route('/live')
def live_page():
    sid = str(uuid.uuid4())
    return render_template('live.html', session_id=sid)


@app.route('/api/live_frame', methods=['POST'])
def live_frame():
    data = request.get_json(force=True)
    session_id = data.get('session_id', 'default')
    frame_b64  = data.get('frame')
    if not frame_b64:
        return jsonify({'error': 'No frame'}), 400
    result = process_live_frame(session_id, frame_b64)
    return jsonify(result)


@app.route('/demo')
def demo_page():
    """Generate synthetic demo analysis without a real video."""
    np.random.seed(42)
    n = 80
    t = [round(i * 0.5, 1) for i in range(n)]

    # Simulate: calm → escalation → near-panic → calm
    base = np.concatenate([
        np.random.uniform(5, 20, 25),
        np.linspace(20, 75, 25),
        np.random.uniform(70, 90, 15) + np.random.normal(0, 5, 15),
        np.linspace(75, 25, 15),
    ])
    scores = np.clip(base + np.random.normal(0, 3, n), 0, 100).tolist()

    analyses = []
    for i in range(n):
        sc = round(scores[i], 2)
        analyses.append({
            'frame':       i * 12,
            'timestamp':   t[i],
            'panic_score': sc,
            'avg_speed':   round(sc / 40, 4),
            'entropy':     round(sc / 120, 4),
            'density':     round(sc / 300, 4),
            'acceleration':round(abs(scores[i] - scores[i-1]) / 50 if i > 0 else 0, 4),
            'risk_level':  risk_label(sc),
        })

    max_score = max(scores)
    peak_ts   = t[scores.index(max_score)]
    overall   = risk_label(max_score)

    result = {
        'total_frames_analysed': n,
        'duration_seconds': t[-1],
        'fps': 25.0,
        'max_panic_score': round(max_score, 2),
        'avg_panic_score': round(sum(scores)/n, 2),
        'peak_panic_timestamp': peak_ts,
        'overall_risk': overall,
        'recommendation': recommendation_text(overall, peak_ts),
        'frame_analyses': analyses,
        'key_frames': [],
        'chart': {
            'timestamps':   t,
            'panic_scores': [round(s, 2) for s in scores],
            'speeds':       [round(s/40, 4) for s in scores],
            'densities':    [round(s/300, 4) for s in scores],
            'entropies':    [round(s/120, 4) for s in scores],
        },
        'metrics_summary': {
            'max_speed':   round(max_score/40, 4),
            'max_entropy': round(max_score/120, 4),
            'max_density': round(max_score/300, 4),
            'max_accel':   0.32,
        },
        'is_demo': True,
    }
    return render_template('result.html', result=result, job_id='demo')


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, port=5000, threaded=True)
