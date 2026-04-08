"""Real-time volleyball tracking demo with webcam + upload support."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.detection.ball_detector import BallDetection, BallDetector
from src.tracking.trajectory_predictor import (
    BallTrajectoryPredictor,
    TrajectoryConfig,
    draw_speed_label,
    draw_trajectory_overlay,
)

# ---------------------------------------------------------------------------
# Shared frame processing
# ---------------------------------------------------------------------------

MIN_BALL_DIM = 5.0
MAX_BALL_DIM = 45.0
STUCK_THRESHOLD = 8
MIN_MOVEMENT_PX = 5.0
GRID_SIZE = 30


@dataclass
class FrameStats:
    detected: bool
    total_detections: int
    total_frames: int
    fps: float
    speed_px_s: float | None


class FrameProcessor:
    """Wraps detector + predictor + filters into a single process_frame() call."""

    def __init__(self, model_path: str, conf_threshold: float = 0.2) -> None:
        self.detector = BallDetector(
            model_path=model_path,
            class_id=0,
            conf_threshold=conf_threshold,
        )
        self.config = TrajectoryConfig(
            process_noise=1e-2,
            measurement_noise=5e-1,
            past_trail_length=40,
            future_frames=20,
        )
        self.predictor = BallTrajectoryPredictor(config=self.config)

        # Tracking state
        self.prev_centroid: tuple[float, float] | None = None
        self.stuck_count = 0
        self.total_detections = 0
        self.total_frames = 0
        self.max_step_px = 150.0

        # Grid-based motion tracking
        self.det_streak: dict[int, int] = {}
        self.det_last_frame: dict[int, int] = {}

    def _grid_key(self, cx: float, cy: float) -> int:
        return int(cx // GRID_SIZE) * 10000 + int(cy // GRID_SIZE)

    def _is_stuck(self, cx: float, cy: float) -> bool:
        key = self._grid_key(cx, cy)
        return self.det_streak.get(key, 0) >= STUCK_THRESHOLD

    def _record_detections(self, dets: list[BallDetection]) -> None:
        for det in dets:
            cx = (det.x1 + det.x2) / 2
            cy = (det.y1 + det.y2) / 2
            key = self._grid_key(cx, cy)
            last = self.det_last_frame.get(key, -999)
            if self.total_frames - last <= 2:
                self.det_streak[key] = self.det_streak.get(key, 0) + 1
            else:
                self.det_streak[key] = 1
            self.det_last_frame[key] = self.total_frames

    def _pick_moving(self, dets: list[BallDetection]) -> BallDetection | None:
        moving = [d for d in dets if not self._is_stuck(
            (d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2
        )]
        if moving:
            return max(moving, key=lambda d: d.confidence)
        return max(dets, key=lambda d: d.confidence) if dets else None

    def reset(self) -> None:
        """Reset all tracking state."""
        self.predictor.reset()
        self.prev_centroid = None
        self.stuck_count = 0
        self.total_detections = 0
        self.total_frames = 0
        self.det_streak.clear()
        self.det_last_frame.clear()

    def process_frame(self, frame: np.ndarray, fps: float = 30.0) -> tuple[np.ndarray, FrameStats]:
        """Process a single frame: detect, track, predict, draw.

        Returns annotated frame and stats.
        """
        t0 = time.time()
        self.total_frames += 1

        # Detect
        raw_dets = self.detector.detect_frame(frame)

        # Size filter
        dets = [
            d for d in raw_dets
            if MIN_BALL_DIM <= (d.x2 - d.x1) <= MAX_BALL_DIM
            and MIN_BALL_DIM <= (d.y2 - d.y1) <= MAX_BALL_DIM
        ]

        # Record for motion analysis
        self._record_detections(dets)

        # Pick best detection
        best_det: BallDetection | None = None
        if dets:
            if self.prev_centroid is None or self.stuck_count >= STUCK_THRESHOLD:
                if self.stuck_count >= STUCK_THRESHOLD:
                    self.predictor.reset()
                    self.prev_centroid = None
                    self.stuck_count = 0
                best_det = self._pick_moving(dets)
            else:
                for det in sorted(dets, key=lambda d: d.confidence, reverse=True):
                    cx = (det.x1 + det.x2) / 2
                    cy = (det.y1 + det.y2) / 2
                    dist = float(np.hypot(cx - self.prev_centroid[0], cy - self.prev_centroid[1]))
                    if dist <= self.max_step_px:
                        best_det = det
                        break

        # Check stuck
        if best_det is not None and self.prev_centroid is not None:
            cx = (best_det.x1 + best_det.x2) / 2
            cy = (best_det.y1 + best_det.y2) / 2
            movement = float(np.hypot(cx - self.prev_centroid[0], cy - self.prev_centroid[1]))
            if movement < MIN_MOVEMENT_PX:
                self.stuck_count += 1
            else:
                self.stuck_count = 0

        # Extract centroid + bbox
        centroid: tuple[int, int] | None = None
        bbox: tuple[int, int, int, int] | None = None
        if best_det is not None:
            cx = int((best_det.x1 + best_det.x2) / 2)
            cy = int((best_det.y1 + best_det.y2) / 2)
            centroid = (cx, cy)
            bbox = (int(best_det.x1), int(best_det.y1), int(best_det.x2), int(best_det.y2))
            self.prev_centroid = (float(cx), float(cy))
            self.total_detections += 1

        # Update kalman + predict
        self.predictor.update(centroid)
        future = self.predictor.predict_future()

        # Draw overlay
        annotated = frame.copy()
        draw_trajectory_overlay(annotated, bbox, self.predictor.past_positions, future, self.config)

        speed = self.predictor.current_speed_px
        draw_speed_label(annotated, speed, fps, position=(20, 40))

        elapsed = time.time() - t0
        actual_fps = 1.0 / elapsed if elapsed > 0 else 0.0

        stats = FrameStats(
            detected=best_det is not None,
            total_detections=self.total_detections,
            total_frames=self.total_frames,
            fps=actual_fps,
            speed_px_s=speed * fps if speed else None,
        )

        return annotated, stats


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


def _get_model_path() -> str:
    """Find the best available model."""
    candidates = [
        Path("models/volleyball_ball_yolo/weights/best.pt"),
        Path("models/volleyball_ball_yolo_v3/weights/best.pt"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(candidates[0])


def webcam_tab() -> None:
    """Live webcam tracking tab."""
    st.subheader("Live Webcam Ball Tracking")
    st.caption("Point a camera at a volleyball game for real-time detection + trajectory prediction.")

    col1, col2 = st.columns([3, 1])

    with col2:
        model_path = st.text_input("Model path", value=_get_model_path(), key="wc_model")
        conf = st.slider("Confidence", 0.05, 0.5, 0.2, 0.05, key="wc_conf")
        camera_idx = st.number_input("Camera index", value=0, min_value=0, max_value=5, key="wc_cam")

    with col1:
        start_btn = st.button("Start Camera", type="primary", key="wc_start")
        stop_btn = st.button("Stop Camera", key="wc_stop")

        video_placeholder = st.empty()
        stats_placeholder = st.empty()

    if start_btn:
        if not Path(model_path).exists():
            st.error(f"Model not found: {model_path}")
            return

        processor = FrameProcessor(model_path=model_path, conf_threshold=conf)
        cap = cv2.VideoCapture(int(camera_idx))

        if not cap.isOpened():
            st.error(f"Cannot open camera {camera_idx}. Check permissions or try a different index.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        st.session_state["webcam_running"] = True

        while st.session_state.get("webcam_running", False):
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera feed lost.")
                break

            annotated, stats = processor.process_frame(frame, fps)

            # Convert BGR to RGB for streamlit
            video_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

            det_rate = 100 * stats.total_detections / stats.total_frames if stats.total_frames else 0
            stats_placeholder.markdown(
                f"**FPS:** {stats.fps:.0f} | "
                f"**Detections:** {stats.total_detections}/{stats.total_frames} ({det_rate:.0f}%) | "
                f"**Speed:** {stats.speed_px_s:.0f} px/s" if stats.speed_px_s else
                f"**FPS:** {stats.fps:.0f} | "
                f"**Detections:** {stats.total_detections}/{stats.total_frames} ({det_rate:.0f}%)"
            )

        cap.release()

    if stop_btn:
        st.session_state["webcam_running"] = False


def upload_tab() -> None:
    """Upload + process video tab."""
    st.subheader("Upload a Volleyball Video")
    st.caption("Upload a side-angle volleyball clip and see ball tracking + trajectory prediction.")

    model_path = st.text_input("Model path", value=_get_model_path(), key="up_model")
    conf = st.slider("Confidence", 0.05, 0.5, 0.2, 0.05, key="up_conf")

    uploaded = st.file_uploader("Drop a volleyball clip", type=["mp4", "mov", "m4v"], key="up_file")

    if uploaded and st.button("Analyze", type="primary", key="up_go"):
        if not Path(model_path).exists():
            st.error(f"Model not found: {model_path}")
            return

        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        processor = FrameProcessor(model_path=model_path, conf_threshold=conf)
        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            st.error("Cannot open uploaded video.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Process to output video
        out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress = st.progress(0, text="Processing...")
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, stats = processor.process_frame(frame, fps)
            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 10 == 0 or frame_idx == total_frames:
                pct = frame_idx / total_frames if total_frames else 0
                det_rate = 100 * stats.total_detections / stats.total_frames if stats.total_frames else 0
                progress.progress(
                    min(pct, 1.0),
                    text=f"Frame {frame_idx}/{total_frames} -- {det_rate:.0f}% detection rate",
                )

        cap.release()
        writer.release()
        progress.progress(1.0, text="Done!")

        # Show results
        det_rate = 100 * processor.total_detections / processor.total_frames if processor.total_frames else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Frames", f"{processor.total_frames}")
        col2.metric("Detections", f"{processor.total_detections}")
        col3.metric("Detection Rate", f"{det_rate:.1f}%")

        st.video(out_path)

        # Download button
        with open(out_path, "rb") as f:
            st.download_button(
                "Download Annotated Video",
                data=f.read(),
                file_name=f"tracked_{uploaded.name}",
                mime="video/mp4",
            )


def gallery_tab() -> None:
    """Show pre-processed demo videos."""
    st.subheader("Demo Gallery")
    st.caption("Pre-processed volleyball clips with trajectory tracking.")

    runs_dir = Path("data/processed/runs")
    if not runs_dir.exists():
        st.info("No processed videos found. Upload a clip in the Upload tab first.")
        return

    videos = sorted(runs_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not videos:
        st.info("No processed videos found.")
        return

    selected = st.selectbox(
        "Select a demo video",
        options=videos,
        format_func=lambda p: p.name,
    )

    if selected:
        st.video(str(selected))
        size_mb = selected.stat().st_size / 1024 / 1024
        st.caption(f"{selected.name} ({size_mb:.1f} MB)")


def main() -> None:
    st.set_page_config(
        page_title="VBCV -- Volleyball Tracker",
        page_icon="🏐",
        layout="wide",
    )

    st.title("Volleyball Ball Tracker")
    st.caption("Real-time ball detection + trajectory prediction using YOLOv8 + Kalman filter")

    tab_webcam, tab_upload, tab_gallery = st.tabs(["Live Webcam", "Upload Video", "Gallery"])

    with tab_webcam:
        webcam_tab()

    with tab_upload:
        upload_tab()

    with tab_gallery:
        gallery_tab()


if __name__ == "__main__":
    main()
