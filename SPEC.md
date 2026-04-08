# VBCV Spec (Sprint 0)

## Objective
Build a computer-vision pipeline for volleyball serve analysis from consumer video.

## Locked Input Format (updated Sprint 0 → Sprint 8)
- Source: smartphone or broadcast video recorded in 1080p at 30-60 fps.
- Capture setup: tripod-mounted or fixed broadcast camera.
- Camera angle: side-court / broadcast view (elevated sideline perspective preferred).
- Shot framing: full court visible with net, server, and receiving court in frame.
- File format: `.mp4` with H.264 video codec.

## Why this format
- Side/broadcast angle is the standard in volleyball footage and has the most available training data.
- Pretrained volleyball detection models (YOLOv8x, VolleyVision) are trained on broadcast angles.
- Court lines are clearly visible from side view, improving homography calibration.
- Tripod/fixed camera stabilizes frame-to-frame geometry for court calibration and speed estimates.
- 60 fps improves contact and launch-frame timing for velocity estimation (30 fps minimum).

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
- Court lines are sufficiently visible for calibration (side-view makes this easier).
- Camera is static for each clip (tripod or fixed broadcast mount).
- Lighting is adequate to keep motion blur manageable.
- Side/broadcast angle introduces depth ambiguity on the Z-axis, so speed and landing estimates are 2D court-plane projections (not true 3D).

## Out of Scope (Sprint 0)
- Multi-camera fusion
- Real-time edge deployment
- Multi-ball or rally tracking
