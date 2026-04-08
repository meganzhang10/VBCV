# VBCV Spec (Sprint 0)

## Objective
Build a computer-vision pipeline for volleyball serve analysis from consumer video.

## Locked Input Format (Sprint 0 decision)
- Source: smartphone video recorded in 1080p at 60 fps (preferred), minimum 30 fps.
- Capture setup: tripod-mounted camera.
- Camera angle: behind-server baseline view, centered as much as possible.
- Shot framing: full server body, ball toss, contact point, net, and receiving court visible.
- File format: `.mp4` with H.264 video codec.

## Why this format
- Behind-server minimizes left/right depth ambiguity for serve trajectory.
- Tripod stabilizes frame-to-frame geometry for court calibration and speed estimates.
- 60 fps improves contact and launch-frame timing for velocity estimation.

## Tooling
- Language: Python 3.11+
- CV: OpenCV
- Detection: Ultralytics YOLO
- Data/Math: NumPy, Pandas, SciPy
- Dashboard (phase 1): Streamlit
- Testing: Pytest
- Lint/format/type: Ruff + mypy

## Sprint 0 Scope
- Repository layout and starter modules created.
- Basic project assumptions encoded in `src/config.py`.
- Sample-video folder and manifest created.
- Tests validating structural assumptions.

## Phase Metrics (planned)
- Ball detection precision/recall
- Tracking continuity (% frames with valid ball track)
- Speed estimate stability (frame-to-frame variance)
- Landing point localization error (pixels, then court units)
- Serve session stats:
  - Average speed
  - Max speed
  - In/out ratio
  - Landing heatmap density

## Assumptions
- Exactly one ball is relevant per clip.
- Server is in frame before toss.
- Court lines are sufficiently visible for calibration.
- Camera is static for each clip.
- Lighting is adequate to keep motion blur manageable.

## Out of Scope (Sprint 0)
- Multi-camera fusion
- Real-time edge deployment
- Multi-ball or rally tracking
