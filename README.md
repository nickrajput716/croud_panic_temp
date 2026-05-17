# Crowd Panic Prediction System

**Public Safety AI** — Real-time stampede risk detection using computer vision.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Flask server
python app.py

# 3. Open browser
http://localhost:5000
```

## Features

| Feature | Details |
|---|---|
| Video Upload | MP4, AVI, MOV, MKV, WEBM, FLV (up to 200 MB) |
| Live Webcam | Real-time frame-by-frame analysis via browser camera |
| Demo Mode | Synthetic analysis report — no video needed |
| Annotated Frames | Optical flow arrows overlaid on up to 12 key frames |
| JSON Export | Full frame-by-frame data downloadable as JSON |

## Computer Vision Pipeline

```
Frame N ──► Resize 640×360
         ├─► MOG2 Background Subtractor ──► Crowd Density
         ├─► Farneback Optical Flow ──────► Motion Vectors
         │     ├─► Magnitude ──────────────► Speed + Turbulence
         │     ├─► Arctan2 + Histogram ──── Direction Entropy
         │     └─► Gradient (∇·F) ────────► Flow Divergence
         └─► Weighted Fusion ────────────► Panic Score 0–100
```

## Panic Score Formula

| Signal | Weight | Description |
|---|---|---|
| Avg Speed | 20% | Mean optical-flow magnitude |
| Direction Entropy | 30% | Chaos in movement directions |
| Crowd Density | 12% | MOG2 foreground pixel ratio |
| Acceleration | 15% | Sudden speed change |
| Turbulence | 13% | Spatial variance of speeds |
| Flow Divergence | 10% | Outward scatter (escape pattern) |

## Risk Levels

| Level | Score | Action |
|---|---|---|
| NORMAL | 0–24 | Routine monitoring |
| LOW | 25–44 | Increase frequency |
| MODERATE | 45–64 | Deploy personnel |
| HIGH | 65–79 | Initiate dispersal |
| CRITICAL | 80–100 | Emergency response |

## Architecture

- **Backend:** Python 3.x + Flask + OpenCV + NumPy
- **Processing:** Background thread per upload job
- **Live Mode:** WebRTC webcam capture → base64 JPEG → Flask API → OpenCV → annotated JPEG
- **Frontend:** Plain HTML + Chart.js (CDN) — no CSS framework
