"""Single-ball tracking for Sprint 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detection.ball_detector import BallDetection, BallDetector


@dataclass
class TrackPoint:
    frame_index: int
    timestamp_sec: float
    detected: bool
    interpolated: bool
    raw_x: float | None
    raw_y: float | None
    smooth_x: float | None = None
    smooth_y: float | None = None
    confidence: float | None = None


def _center_xy(detection: BallDetection) -> tuple[float, float]:
    return ((detection.x1 + detection.x2) / 2.0, (detection.y1 + detection.y2) / 2.0)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def select_detection(
    detections: list[BallDetection],
    previous_xy: tuple[float, float] | None,
    max_step_px: float,
) -> BallDetection | None:
    """Select detection for current frame using nearest-neighbor association."""
    if not detections:
        return None
    if previous_xy is None:
        return max(detections, key=lambda d: d.confidence)

    nearest = min(detections, key=lambda d: _distance(_center_xy(d), previous_xy))
    if _distance(_center_xy(nearest), previous_xy) <= max_step_px:
        return nearest
    return None


def run_tracking(
    detector: BallDetector,
    input_video: str | Path,
    max_step_px: float = 120.0,
) -> tuple[list[TrackPoint], float]:
    """Track ball center per frame from detector output."""
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    points: list[TrackPoint] = []
    previous_xy: tuple[float, float] | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect_frame(frame)
        selected = select_detection(detections, previous_xy, max_step_px=max_step_px)
        if selected is not None:
            x, y = _center_xy(selected)
            previous_xy = (x, y)
            points.append(
                TrackPoint(
                    frame_index=frame_idx,
                    timestamp_sec=frame_idx / fps,
                    detected=True,
                    interpolated=False,
                    raw_x=x,
                    raw_y=y,
                    confidence=selected.confidence,
                )
            )
        else:
            points.append(
                TrackPoint(
                    frame_index=frame_idx,
                    timestamp_sec=frame_idx / fps,
                    detected=False,
                    interpolated=False,
                    raw_x=None,
                    raw_y=None,
                    confidence=None,
                )
            )
        frame_idx += 1

    cap.release()
    return points, fps


def interpolate_missing(points: list[TrackPoint], max_gap: int = 5) -> None:
    """Fill short detection gaps with linear interpolation."""
    i = 0
    total = len(points)
    while i < total:
        if points[i].raw_x is not None:
            i += 1
            continue
        gap_start = i
        while i < total and points[i].raw_x is None:
            i += 1
        gap_end = i - 1
        prev_idx = gap_start - 1
        next_idx = i
        gap_len = gap_end - gap_start + 1
        if prev_idx < 0 or next_idx >= total:
            continue
        prev_point = points[prev_idx]
        next_point = points[next_idx]
        if prev_point.raw_x is None or next_point.raw_x is None:
            continue
        if gap_len > max_gap:
            continue

        assert prev_point.raw_y is not None and next_point.raw_y is not None
        for j in range(gap_len):
            alpha = (j + 1) / (gap_len + 1)
            interp_x = prev_point.raw_x + alpha * (next_point.raw_x - prev_point.raw_x)
            interp_y = prev_point.raw_y + alpha * (next_point.raw_y - prev_point.raw_y)
            p = points[gap_start + j]
            p.raw_x = interp_x
            p.raw_y = interp_y
            p.interpolated = True


def smooth_track(points: list[TrackPoint], window_size: int = 5) -> None:
    """Apply moving-average smoothing to tracked coordinates."""
    if window_size <= 1:
        for p in points:
            p.smooth_x = p.raw_x
            p.smooth_y = p.raw_y
        return

    radius = window_size // 2
    for i, point in enumerate(points):
        if point.raw_x is None or point.raw_y is None:
            point.smooth_x = None
            point.smooth_y = None
            continue

        xs: list[float] = []
        ys: list[float] = []
        start = max(0, i - radius)
        end = min(len(points), i + radius + 1)
        for j in range(start, end):
            neighbor = points[j]
            if neighbor.raw_x is None or neighbor.raw_y is None:
                continue
            xs.append(neighbor.raw_x)
            ys.append(neighbor.raw_y)
        if xs and ys:
            point.smooth_x = float(np.mean(xs))
            point.smooth_y = float(np.mean(ys))
        else:
            point.smooth_x = point.raw_x
            point.smooth_y = point.raw_y


def write_track_csv(points: list[TrackPoint], out_csv: str | Path) -> None:
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(p) for p in points]).to_csv(out_path, index=False)


def write_track_json(points: list[TrackPoint], out_json: str | Path) -> None:
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(p) for p in points]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_trail_video(
    input_video: str | Path,
    output_video: str | Path,
    points: list[TrackPoint],
    trail_length: int = 30,
) -> None:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps,
        (width, height),
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        start = max(0, frame_idx - trail_length + 1)
        trail_points = points[start : frame_idx + 1]

        previous_draw: tuple[int, int] | None = None
        for p in trail_points:
            draw_x = p.smooth_x if p.smooth_x is not None else p.raw_x
            draw_y = p.smooth_y if p.smooth_y is not None else p.raw_y
            if draw_x is None or draw_y is None:
                previous_draw = None
                continue
            draw_xy = (int(draw_x), int(draw_y))
            if previous_draw is not None:
                cv2.line(frame, previous_draw, draw_xy, (0, 220, 255), 2)
            previous_draw = draw_xy

        if frame_idx < len(points):
            current = points[frame_idx]
            current_x = current.smooth_x if current.smooth_x is not None else current.raw_x
            current_y = current.smooth_y if current.smooth_y is not None else current.raw_y
            if current_x is not None and current_y is not None:
                cv2.circle(frame, (int(current_x), int(current_y)), 5, (0, 255, 0), -1)
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
