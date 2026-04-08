"""Volleyball Ball Tracker -- Gradio Demo App.

Upload a volleyball video or use your webcam to see real-time ball detection
+ trajectory prediction using YOLOv8 and a Kalman filter.

Usage:
    cd /tmp/VBCV
    python3 -m src.visualization.demo_gradio
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.ball_detector import BallDetection, BallDetector
from src.tracking.trajectory_predictor import (
    BallTrajectoryPredictor,
    TrajectoryConfig,
    draw_speed_label,
    draw_trajectory_overlay,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_BALL_DIM = 5.0
MAX_BALL_DIM = 45.0
STUCK_THRESHOLD = 20
MIN_MOVEMENT_PX = 3.0
GRID_SIZE = 30


def _find_model() -> tuple[str, int]:
    """Return (model_path, class_id). Prefers custom model, falls back to COCO."""
    custom = PROJECT_ROOT / "models" / "volleyball_ball_yolo" / "weights" / "best.pt"
    if custom.exists():
        return str(custom), 0
    # Fall back to COCO pretrained (class 32 = sports ball)
    return "yolov8n.pt", 32


# ---------------------------------------------------------------------------
# Frame processor (same logic as run_trajectory_demo.py, packaged cleanly)
# ---------------------------------------------------------------------------

class Processor:
    def __init__(self, model_path: str, class_id: int = 0, conf: float = 0.2) -> None:
        self.detector = BallDetector(model_path=model_path, class_id=class_id, conf_threshold=conf)
        self.config = TrajectoryConfig(
            process_noise=1e-2,
            measurement_noise=5e-1,
            past_trail_length=40,
            future_frames=20,
        )
        self.predictor = BallTrajectoryPredictor(config=self.config)
        self.prev: tuple[float, float] | None = None
        self.stuck = 0
        self.det_count = 0
        self.frame_count = 0
        self.det_streak: dict[int, int] = {}
        self.det_last: dict[int, int] = {}

    def _gk(self, cx: float, cy: float) -> int:
        return int(cx // GRID_SIZE) * 10000 + int(cy // GRID_SIZE)

    def _is_stuck(self, cx: float, cy: float) -> bool:
        return self.det_streak.get(self._gk(cx, cy), 0) >= STUCK_THRESHOLD

    def _record(self, dets: list[BallDetection]) -> None:
        for d in dets:
            cx, cy = (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
            k = self._gk(cx, cy)
            if self.frame_count - self.det_last.get(k, -999) <= 2:
                self.det_streak[k] = self.det_streak.get(k, 0) + 1
            else:
                self.det_streak[k] = 1
            self.det_last[k] = self.frame_count

    def _pick_moving(self, dets: list[BallDetection]) -> BallDetection | None:
        moving = [d for d in dets if not self._is_stuck((d.x1+d.x2)/2, (d.y1+d.y2)/2)]
        pool = moving or dets
        return max(pool, key=lambda d: d.confidence) if pool else None

    def process(self, frame: np.ndarray, fps: float = 30.0) -> np.ndarray:
        self.frame_count += 1

        raw = self.detector.detect_frame(frame)
        dets = [d for d in raw
                if MIN_BALL_DIM <= (d.x2-d.x1) <= MAX_BALL_DIM
                and MIN_BALL_DIM <= (d.y2-d.y1) <= MAX_BALL_DIM]

        self._record(dets)

        best: BallDetection | None = None
        if dets:
            if self.prev is None or self.stuck >= STUCK_THRESHOLD:
                if self.stuck >= STUCK_THRESHOLD:
                    self.predictor.reset()
                    self.prev = None
                    self.stuck = 0
                best = self._pick_moving(dets)
            else:
                missed = self.predictor.frames_since_detection
                max_dist = 150 + (missed * 30)
                for d in sorted(dets, key=lambda x: x.confidence, reverse=True):
                    cx, cy = (d.x1+d.x2)/2, (d.y1+d.y2)/2
                    if float(np.hypot(cx-self.prev[0], cy-self.prev[1])) <= max_dist:
                        best = d
                        break
                if best is None and missed > 5:
                    self.predictor.reset()
                    self.prev = None
                    best = self._pick_moving(dets)

        if best and self.prev:
            cx, cy = (best.x1+best.x2)/2, (best.y1+best.y2)/2
            if float(np.hypot(cx-self.prev[0], cy-self.prev[1])) < MIN_MOVEMENT_PX:
                self.stuck += 1
            else:
                self.stuck = 0

        centroid = None
        bbox = None
        if best:
            cx = int((best.x1+best.x2)/2)
            cy = int((best.y1+best.y2)/2)
            centroid = (cx, cy)
            bbox = (int(best.x1), int(best.y1), int(best.x2), int(best.y2))
            self.prev = (float(cx), float(cy))
            self.det_count += 1

        self.predictor.update(centroid)
        future = self.predictor.predict_future()

        out = frame.copy()
        draw_trajectory_overlay(out, bbox, self.predictor.past_positions, future, self.config)
        draw_speed_label(out, self.predictor.current_speed_px, fps)

        # Stats overlay
        rate = 100 * self.det_count / self.frame_count if self.frame_count else 0
        cv2.putText(out, f"Detection: {rate:.0f}%  |  Frame {self.frame_count}",
                    (20, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        return out


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

_processor: Processor | None = None


def _get_processor() -> Processor:
    global _processor
    if _processor is None:
        model_path, class_id = _find_model()
    _processor = Processor(model_path=model_path, class_id=class_id, conf=0.15)
    return _processor


def process_video(video_path: str) -> str:
    """Process an uploaded video and return the annotated video path."""
    global _processor
    model_path, class_id = _find_model()
    _processor = Processor(model_path=model_path, class_id=class_id, conf=0.15)
    proc = _processor

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = proc.process(frame, fps)
        writer.write(annotated)

    cap.release()
    writer.release()

    rate = 100 * proc.det_count / proc.frame_count if proc.frame_count else 0
    print(f"Processed {proc.frame_count} frames, {proc.det_count} detections ({rate:.1f}%)")

    return out_path


def process_webcam_frame(frame: np.ndarray) -> np.ndarray:
    """Process a single webcam frame for real-time tracking."""
    if frame is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    proc = _get_processor()
    # Gradio webcam gives RGB, our pipeline uses BGR
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated = proc.process(bgr, 30.0)
    # Convert back to RGB for gradio
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="VBCV -- Volleyball Ball Tracker") as app:
        gr.Markdown(
            "# Volleyball Ball Tracker\n"
            "Real-time ball detection + trajectory prediction using **YOLOv8** and a **Kalman filter**.\n\n"
            "Upload a side-angle volleyball video to see ball tracking with predicted trajectory overlay."
        )

        with gr.Tabs():
            with gr.TabItem("Upload Video"):
                gr.Markdown("Upload a volleyball clip (side/broadcast angle works best).")
                with gr.Row():
                    video_input = gr.Video(label="Input Video")
                    video_output = gr.Video(label="Tracked Output")
                analyze_btn = gr.Button("Analyze", variant="primary", size="lg")
                analyze_btn.click(fn=process_video, inputs=video_input, outputs=video_output)

            with gr.TabItem("Live Webcam"):
                gr.Markdown(
                    "Point your webcam at a volleyball game or a TV showing a match.\n\n"
                    "The tracker will detect the ball and show the predicted trajectory in real-time."
                )
                webcam = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed")
                webcam.stream(fn=process_webcam_frame, inputs=webcam, outputs=webcam)

            with gr.TabItem("About"):
                gr.Markdown(
                    "## How it works\n\n"
                    "1. **YOLOv8** detects the volleyball in each frame (trained on 15k+ volleyball images)\n"
                    "2. **Size filtering** removes false positives (player heads, jerseys) -- "
                    "real volleyballs are 10-40px at broadcast distance\n"
                    "3. **Motion filtering** ignores stationary balls on the ground\n"
                    "4. **Kalman filter** (6D: position + velocity + acceleration) smooths the track "
                    "and predicts the future trajectory\n\n"
                    "### Visual overlay\n"
                    "- **Green box**: detected ball position\n"
                    "- **Yellow trail**: past ball path (last 40 frames)\n"
                    "- **Orange dots**: predicted future trajectory (next 20 frames)\n"
                    "- **White dot**: ball centroid\n\n"
                    "### Tech stack\n"
                    "Python, OpenCV, Ultralytics YOLOv8, NumPy, Gradio\n\n"
                    "Built for UC Berkeley volleyball serve analysis research."
                )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_port=7860, share=True)
