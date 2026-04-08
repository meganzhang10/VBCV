"""Run ball detection + Kalman trajectory prediction on a video.

Produces a video with:
- Bounding box on detected ball (yellow)
- Solid fading trail for past positions (gray -> cyan)
- Dotted line for predicted future trajectory (red/orange, fading)
- Glow effect on ball centroid

Usage:
    python3 -m scripts.run_trajectory_demo \
        --input-video data/raw/videos/side_angle/volleyball_research_clip.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.detection.ball_detector import BallDetection, BallDetector
from src.tracking.trajectory_predictor import (
    BallTrajectoryPredictor,
    TrajectoryConfig,
    draw_speed_label,
    draw_trajectory_overlay,
)


def run_demo(
    input_video: Path,
    output_video: Path,
    model_path: Path,
    conf_threshold: float = 0.1,
    max_step_px: float = 150.0,
) -> None:
    # If using COCO pretrained model (yolov8n.pt), use class 32 (sports ball)
    # If using custom volleyball model, use class 0
    is_coco = "yolov8" in model_path.name and "volleyball" not in model_path.name
    class_id = 32 if is_coco else 0

    detector = BallDetector(
        model_path=str(model_path),
        class_id=class_id,
        conf_threshold=conf_threshold,
    )

    config = TrajectoryConfig(
        process_noise=1e-2,       # lower = smoother, less reactive to noise
        measurement_noise=5e-1,   # higher = trust predictions over noisy detections
        past_trail_length=40,
        future_frames=20,
    )
    predictor = BallTrajectoryPredictor(config=config)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    print(f"Input:  {input_video} ({width}x{height} @ {fps:.0f}fps, {total_frames} frames)")
    print(f"Output: {output_video}")
    print(f"Model:  {model_path}")
    print()

    prev_centroid: tuple[float, float] | None = None
    frame_idx = 0
    detected_count = 0

    # Motion-based ball selection:
    # Track how long our currently-tracked ball has been sitting still.
    # If it hasn't moved in N frames, it's stationary -- drop it and
    # look for a detection that IS moving.
    stuck_count = 0  # how many frames our tracked ball hasn't moved
    stuck_threshold = 20  # after this many stuck frames, reset tracking (was 8, too aggressive)
    min_movement_px = 3.0  # less than this between frames = "not moving"

    # Keep a per-detection motion history to find moving balls
    # Maps grid_key -> list of (frame_idx, cx, cy)
    det_history: dict[int, list[tuple[int, float, float]]] = {}
    grid_size = 30

    def _grid_key(cx: float, cy: float) -> int:
        return int(cx // grid_size) * 10000 + int(cy // grid_size)

    def _record_detections(dets: list[BallDetection], current_frame: int) -> None:
        for det in dets:
            cx = (det.x1 + det.x2) / 2
            cy = (det.y1 + det.y2) / 2
            key = _grid_key(cx, cy)
            if key not in det_history:
                det_history[key] = []
            det_history[key].append((current_frame, cx, cy))
            # Keep last 60 entries per cell
            if len(det_history[key]) > 60:
                det_history[key] = det_history[key][-30:]

    def _detection_is_stuck(det: BallDetection, current_frame: int) -> bool:
        """Check if this detection has been in the same grid cell for a while."""
        cx = (det.x1 + det.x2) / 2
        cy = (det.y1 + det.y2) / 2
        key = _grid_key(cx, cy)
        history = det_history.get(key, [])
        if len(history) < stuck_threshold:
            return False
        recent = [h for h in history if current_frame - h[0] <= stuck_threshold + 2]
        return len(recent) >= stuck_threshold

    def _pick_most_moving(dets: list[BallDetection], current_frame: int) -> BallDetection | None:
        """Among all detections, pick the one that's NOT stuck in one place."""
        # Prefer detections that aren't stuck
        moving = [d for d in dets if not _detection_is_stuck(d, current_frame)]
        if moving:
            return max(moving, key=lambda d: d.confidence)
        # Everything looks stuck -- just pick highest confidence
        return max(dets, key=lambda d: d.confidence) if dets else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Detect ball
        raw_detections = detector.detect_frame(frame)

        # Size filter: real volleyballs in broadcast footage are small (10-40px)
        # Anything bigger is a head, jersey, hand, or other false positive
        min_ball_dim = 5.0
        max_ball_dim = 45.0
        detections = [
            d for d in raw_detections
            if min_ball_dim <= (d.x2 - d.x1) <= max_ball_dim
            and min_ball_dim <= (d.y2 - d.y1) <= max_ball_dim
        ]

        # Record all detections for motion analysis
        _record_detections(detections, frame_idx)

        best_det = None
        if detections:
            if prev_centroid is None or stuck_count >= stuck_threshold:
                # No lock yet, or our current target is stuck -- find a moving ball
                if stuck_count >= stuck_threshold:
                    predictor.reset()
                    prev_centroid = None
                    stuck_count = 0
                best_det = _pick_most_moving(detections, frame_idx)
            else:
                # We have a lock -- find nearest detection
                # Scale max_step based on how many frames we've missed
                # After a spike, ball can travel far before we redetect
                frames_missed = predictor.frames_since_detection
                effective_max_step = max_step_px + (frames_missed * 30)  # 30px per missed frame

                candidates = sorted(detections, key=lambda d: d.confidence, reverse=True)
                for det in candidates:
                    cx = (det.x1 + det.x2) / 2
                    cy = (det.y1 + det.y2) / 2
                    dist = float(np.hypot(cx - prev_centroid[0], cy - prev_centroid[1]))
                    if dist <= effective_max_step:
                        best_det = det
                        break

                # If nothing matched within range but we've missed many frames,
                # just grab the best moving detection and start fresh
                if best_det is None and frames_missed > 5:
                    predictor.reset()
                    prev_centroid = None
                    best_det = _pick_most_moving(detections, frame_idx)

        # Check if our tracked ball is actually moving
        if best_det is not None and prev_centroid is not None:
            cx = (best_det.x1 + best_det.x2) / 2
            cy = (best_det.y1 + best_det.y2) / 2
            movement = float(np.hypot(cx - prev_centroid[0], cy - prev_centroid[1]))
            if movement < min_movement_px:
                stuck_count += 1
            else:
                stuck_count = 0
        elif best_det is None:
            # No detection this frame -- don't increment stuck (could just be a miss)
            pass

        # Extract centroid and bbox
        centroid: tuple[int, int] | None = None
        bbox: tuple[int, int, int, int] | None = None
        if best_det is not None:
            cx = int((best_det.x1 + best_det.x2) / 2)
            cy = int((best_det.y1 + best_det.y2) / 2)
            centroid = (cx, cy)
            bbox = (int(best_det.x1), int(best_det.y1), int(best_det.x2), int(best_det.y2))
            prev_centroid = (float(cx), float(cy))
            detected_count += 1

        # Update Kalman filter
        predictor.update(centroid)

        # Predict future
        future = predictor.predict_future()

        # Draw everything
        draw_trajectory_overlay(frame, bbox, predictor.past_positions, future, config)
        draw_speed_label(frame, predictor.current_speed_px, fps, position=(20, 40))

        # Frame counter
        cv2.putText(
            frame,
            f"Frame {frame_idx}/{total_frames} | Detected: {detected_count}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = 100 * detected_count / frame_idx if frame_idx else 0
            print(f"  Frame {frame_idx}/{total_frames} — {detected_count} detections ({pct:.1f}%)")

    writer.release()
    cap.release()

    pct = 100 * detected_count / frame_idx if frame_idx else 0
    print(f"\nDone! {detected_count}/{frame_idx} frames with detections ({pct:.1f}%)")
    print(f"Output: {output_video}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ball trajectory prediction demo")
    parser.add_argument("--input-video", required=True, help="Input video path")
    parser.add_argument(
        "--output-video",
        default=None,
        help="Output video path (auto-generated if not set)",
    )
    parser.add_argument(
        "--model-path",
        default="models/volleyball_ball_yolo/weights/best.pt",
        help="YOLO model weights",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.1, help="Detection confidence")
    args = parser.parse_args()

    input_video = Path(args.input_video)
    if args.output_video:
        output_video = Path(args.output_video)
    else:
        output_video = Path("data/processed/runs") / f"{input_video.stem}_trajectory.mp4"

    run_demo(
        input_video=input_video,
        output_video=output_video,
        model_path=Path(args.model_path),
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()
