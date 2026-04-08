# VBCV

Volleyball serve computer-vision pipeline.

## Sprint 0 status
- Project skeleton initialized.
- Input format and assumptions locked in [SPEC.md](./SPEC.md).
- Sample video directory prepared at `data/raw/videos/`.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Sprint 1: Ball Detector
Goal: train a volleyball-specific detector (YOLO) that works on serve clips.

### 1) Record clips
- Collect 20-50 serve clips (or use broadcast/side-angle footage).
- Keep lighting good and camera stable (tripod or fixed broadcast mount, side-court view).
- Add new clips under `data/raw/videos/` and register them in `data/raw/videos/manifest.csv`.

### 2) Extract frames
```bash
python3 -m src.detection.sprint1 extract \
  --manifest data/raw/videos/manifest.csv \
  --videos-dir data/raw/videos \
  --dataset-root data/processed/ball_dataset \
  --data-yaml data/processed/ball_dataset.yaml \
  --every-n-frames 3 \
  --max-frames-per-clip 300
```

### 3) Label volleyball boxes
- Label images in `data/processed/ball_dataset/images/{train,val,test}`.
- Save YOLO labels to matching paths in `data/processed/ball_dataset/labels/{train,val,test}`.
- Use one class only: `0` = volleyball.

### 4) Train YOLO
```bash
python3 -m src.detection.sprint1 train \
  --data-yaml data/processed/ball_dataset.yaml \
  --base-model yolov8n.pt \
  --imgsz 1280 \
  --epochs 80 \
  --batch 8 \
  --project-dir models \
  --run-name volleyball_ball_yolo
```

### 5) Evaluate on unseen frames
```bash
python3 -m src.detection.sprint1 eval \
  --model-path models/volleyball_ball_yolo/weights/best.pt \
  --images-dir data/processed/ball_dataset/images/test \
  --labels-dir data/processed/ball_dataset/labels/test
```

### 6) Generate demo video
```bash
python3 -m src.detection.sprint1 demo \
  --model-path models/volleyball_ball_yolo/weights/best.pt \
  --input-video data/raw/videos/sample_behind_server_01.mp4 \
  --output-video data/processed/demo_ball_detections.mp4
```

Success target:
- Ball is detected in most frames where visible.
- False positives stay low on unseen clips.

## Sprint 2: Track Ball Across Frames
Goal: convert noisy frame detections into a stable ball path.

Method implemented:
- per-frame detection from trained YOLO model
- nearest-neighbor association between consecutive frames
- short-gap interpolation for missed detections
- moving-average smoothing of trajectory

Run tracking + exports:
```bash
python3 -m src.tracking.sprint2 run \
  --model-path models/volleyball_ball_yolo/weights/best.pt \
  --input-video data/raw/videos/sample_behind_server_01.mp4 \
  --output-video data/processed/ball_track_trail.mp4 \
  --output-csv data/processed/ball_track.csv \
  --output-json data/processed/ball_track.json \
  --max-step-px 120 \
  --max-gap 5 \
  --smooth-window 5 \
  --trail-length 30
```

Deliverables produced:
- `data/processed/ball_track_trail.mp4` (trail overlay video)
- `data/processed/ball_track.csv` (frame-by-frame tracked positions)
- `data/processed/ball_track.json` (same data in JSON)

## Sprint 3: Coordinate Calibration
Goal: convert tracked pixel trajectory into approximate real-world court coordinates.

Method implemented:
- mark known image points (end lines / attack lines / corners)
- estimate homography (image plane -> court plane)
- project tracked points from pixels to court meters
- estimate approximate landing location and 2D court-plane speed
- render top-down court overlay with landing point

### 1) Create calibration template
```bash
python3 -m src.analytics.sprint3 template \
  --output-points data/processed/calibration_points.json
```

Then edit `data/processed/calibration_points.json` and fill each `pixel: [x, y]` from your video frame.

### 2) Run calibration + projection
```bash
python3 -m src.analytics.sprint3 run \
  --points-json data/processed/calibration_points.json \
  --track-csv data/processed/ball_track.csv \
  --output-projected-csv data/processed/ball_track_court.csv \
  --output-report-json data/processed/calibration_report.json \
  --output-overlay-image data/processed/court_overlay_landing.png
```

Outputs:
- `data/processed/ball_track_court.csv` (per-frame court coordinates)
- `data/processed/calibration_report.json` (homography + assumptions + speed/landing)
- `data/processed/court_overlay_landing.png` (top-down court with estimated landing)

Assumptions to keep explicit:
- this is court-plane projection, not full 3D ball arc reconstruction
- landing is approximated as the last valid tracked point in the clip
- speed is approximate 2D speed on the court plane

## Sprint 4: Speed Estimation
Goal: estimate serve speed from calibrated 2D ball motion.

Method implemented:
- use calibrated court coordinates per frame (`court_x_m`, `court_y_m`)
- smooth the court-plane path
- compute instantaneous speed = distance / delta_time
- report:
  - max speed
  - average speed in first 0.3s
- render speed text overlay on video

Run:
```bash
python3 -m src.analytics.sprint4 run \
  --projected-track-csv data/processed/ball_track_court.csv \
  --input-video data/raw/videos/sample_behind_server_01.mp4 \
  --output-speed-csv data/processed/ball_speed.csv \
  --output-summary-json data/processed/serve_speed_summary.json \
  --output-video data/processed/serve_speed_overlay.mp4 \
  --smooth-window 5 \
  --early-window-sec 0.3
```

Outputs:
- `data/processed/ball_speed.csv` (frame-level instantaneous speed)
- `data/processed/serve_speed_summary.json` (max speed + avg first 0.3s)
- `data/processed/serve_speed_overlay.mp4` (video with live estimated speed overlay)

Caveat included in outputs:
- `estimated serve speed from calibrated 2d motion`

## Sprint 5: Trajectory Arc + Landing Detection
Goal: fit a smooth trajectory arc and estimate landing point for useful visuals.

Method implemented:
- fit polynomial arc over tracked ball points
- detect landing using combined heuristics:
  - sudden stop behavior in calibrated motion
  - lowest point near end of track
  - last in-court contact in calibrated coordinates
- render:
  - arc overlay video
  - landing marker on top-down court map

Run:
```bash
python3 -m src.analytics.sprint5 run \
  --projected-track-csv data/processed/ball_track_court.csv \
  --input-video data/raw/videos/sample_behind_server_01.mp4 \
  --output-arc-csv data/processed/ball_trajectory_arc.csv \
  --output-landing-json data/processed/landing_estimate.json \
  --output-video data/processed/trajectory_arc_overlay.mp4 \
  --output-court-map data/processed/landing_court_map.png \
  --arc-degree 2
```

Outputs:
- `data/processed/ball_trajectory_arc.csv` (track + fitted arc points)
- `data/processed/landing_estimate.json` (landing estimate + method notes)
- `data/processed/trajectory_arc_overlay.mp4` (pretty arc overlay)
- `data/processed/landing_court_map.png` (landing marker on court map)

## Sprint 6: Serve Consistency Analytics Dashboard
Goal: aggregate many serves into session-level consistency analytics and visuals.

Metrics implemented:
- average speed
- speed variance
- landing-zone spread
- in/out percentage
- target-zone accuracy
- left/right bias
- depth consistency
- overall consistency score (0-100, weighted composite)

Dashboard visuals:
- landing heatmap
- landing scatter plot
- speed histogram
- per-serve summary table

### 1) Create a session manifest
Create `data/processed/session_serves.csv`:

```csv
serve_id,projected_track_csv
serve_01,data/processed/serve_01_ball_track_court.csv
serve_02,data/processed/serve_02_ball_track_court.csv
serve_03,data/processed/serve_03_ball_track_court.csv
```

`projected_track_csv` paths can be absolute or relative to the manifest location.

### 2) Launch dashboard
```bash
streamlit run src/visualization/dashboard.py -- --manifest data/processed/session_serves.csv
```

The dashboard computes one session across multiple serves and shows summary stats + consistency visualizations.

## Sprint 7: Product Polish Demo
Goal: make the analytics experience feel like a real product demo.

Implemented:
- upload a serve video and auto-process end-to-end
- save serves into named sessions
- side-by-side comparison of two serves
- export annotated outputs as downloadable zip bundles
- cleaner, tabbed Streamlit UI

Demo flow:
1. Open dashboard:
```bash
streamlit run src/visualization/dashboard.py
```
2. In `Upload & Process`:
- set `Session name`
- provide `YOLO model path` and `Calibration points JSON`
- upload `.mp4` serve clip
- click `Process Serve`
3. Output:
- annotated trajectory video preview
- speed + landing metrics
- downloadable annotated zip bundle
4. In `Session Analytics`:
- choose a session
- view summary stats and consistency plots
5. In `Compare Serves`:
- pick two serves from a session
- view side-by-side annotated videos and metric deltas

Notes:
- auto-processing requires a trained YOLO model (`models/.../best.pt`) and filled calibration points JSON.
- all outputs are stored under `data/processed/sessions/<session_id>/<serve_id>/`.

## Sprint 8: Stretch Goal - Target-Zone Coaching
Implemented option 4: user-selected target area with per-serve closeness scoring.

Added:
- per-serve `distance_to_target_m`
- per-serve `target_proximity_score` (0-100)
- `bullseye_hit` flag (within 0.75m of target center)
- coaching summary cards:
  - average miss distance
  - best distance
  - bullseye rate
  - average target score
- coaching ranking chart and feedback note in `Session Analytics`

How to use:
1. Open dashboard and process serves into a session.
2. In `Session Analytics`, set target zone bounds (`x_min/x_max/y_min/y_max`).
3. Review the `Target-Zone Coaching` section for score ranking and coaching feedback.
