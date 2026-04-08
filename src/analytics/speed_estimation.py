"""Speed estimation from calibrated 2D court-plane motion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpeedSummary:
    max_speed_mps: float | None
    max_speed_kph: float | None
    avg_speed_first_0_3s_mps: float | None
    avg_speed_first_0_3s_kph: float | None
    assumption: str


def load_projected_track(projected_track_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(projected_track_csv)
    required = {"frame_index", "timestamp_sec", "court_x_m", "court_y_m"}
    missing = required - set(df.columns)
    if missing:
        missing_joined = ", ".join(sorted(missing))
        raise ValueError(f"Projected track is missing required columns: {missing_joined}")
    return df


def smooth_court_track(track_df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Apply moving-average smoothing on court coordinates."""
    if window_size <= 1:
        output = track_df.copy()
        output["court_x_smooth_m"] = output["court_x_m"]
        output["court_y_smooth_m"] = output["court_y_m"]
        return output

    output = track_df.copy()
    output["court_x_smooth_m"] = (
        output["court_x_m"].rolling(window=window_size, center=True, min_periods=1).mean()
    )
    output["court_y_smooth_m"] = (
        output["court_y_m"].rolling(window=window_size, center=True, min_periods=1).mean()
    )
    return output


def compute_instantaneous_speed(track_df: pd.DataFrame) -> pd.DataFrame:
    """Compute frame-level instantaneous 2D speed in m/s and km/h."""
    output = track_df.copy()
    dx = output["court_x_smooth_m"].diff()
    dy = output["court_y_smooth_m"].diff()
    dt = output["timestamp_sec"].diff()

    distance_m = np.sqrt(dx * dx + dy * dy)
    speed_mps = distance_m / dt.replace(0, np.nan)
    speed_mps = speed_mps.replace([np.inf, -np.inf], np.nan)
    output["inst_speed_mps"] = speed_mps
    output["inst_speed_kph"] = output["inst_speed_mps"] * 3.6
    return output


def summarize_serve_speed(
    speed_df: pd.DataFrame,
    early_window_sec: float = 0.3,
) -> SpeedSummary:
    valid_speed = speed_df.dropna(subset=["inst_speed_mps"])
    if valid_speed.empty:
        return SpeedSummary(
            max_speed_mps=None,
            max_speed_kph=None,
            avg_speed_first_0_3s_mps=None,
            avg_speed_first_0_3s_kph=None,
            assumption="estimated serve speed from calibrated 2d motion",
        )

    max_speed_mps = float(valid_speed["inst_speed_mps"].max())
    first_t = float(valid_speed.iloc[0]["timestamp_sec"])
    early = valid_speed[valid_speed["timestamp_sec"] <= first_t + early_window_sec]
    avg_early_mps = float(early["inst_speed_mps"].mean()) if not early.empty else None

    return SpeedSummary(
        max_speed_mps=max_speed_mps,
        max_speed_kph=max_speed_mps * 3.6,
        avg_speed_first_0_3s_mps=avg_early_mps,
        avg_speed_first_0_3s_kph=None if avg_early_mps is None else avg_early_mps * 3.6,
        assumption="estimated serve speed from calibrated 2d motion",
    )


def export_speed_csv(speed_df: pd.DataFrame, out_csv: str | Path) -> None:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    speed_df.to_csv(path, index=False)


def export_speed_summary_json(summary: SpeedSummary, out_json: str | Path) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def render_speed_overlay_video(
    input_video: str | Path,
    speed_df: pd.DataFrame,
    summary: SpeedSummary,
    output_video: str | Path,
) -> None:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    speed_by_frame: dict[int, float] = {}
    for row in speed_df.itertuples(index=False):
        frame_idx = int(getattr(row, "frame_index"))
        speed = getattr(row, "inst_speed_mps")
        if pd.notna(speed):
            speed_by_frame[frame_idx] = float(speed)

    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        inst = speed_by_frame.get(frame_idx)
        if inst is not None:
            inst_kph = inst * 3.6
            cv2.putText(
                frame,
                f"Estimated 2D Speed: {inst_kph:.1f} km/h",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Estimated 2D Speed: --",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 180, 180),
                2,
                cv2.LINE_AA,
            )

        if summary.max_speed_kph is not None:
            cv2.putText(
                frame,
                f"Max: {summary.max_speed_kph:.1f} km/h",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if summary.avg_speed_first_0_3s_kph is not None:
            cv2.putText(
                frame,
                f"Avg first 0.3s: {summary.avg_speed_first_0_3s_kph:.1f} km/h",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 220, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            "estimated serve speed from calibrated 2d motion",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()


def run_speed_estimation(
    projected_track_csv: str | Path,
    smooth_window: int = 5,
    early_window_sec: float = 0.3,
) -> tuple[pd.DataFrame, SpeedSummary]:
    projected = load_projected_track(projected_track_csv)
    smoothed = smooth_court_track(projected, window_size=smooth_window)
    speed_df = compute_instantaneous_speed(smoothed)
    summary = summarize_serve_speed(speed_df, early_window_sec=early_window_sec)
    return speed_df, summary
