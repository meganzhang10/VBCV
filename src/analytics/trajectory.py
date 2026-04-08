"""Trajectory arc fitting and landing detection for Sprint 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.analytics.court_calibration import draw_topdown_court_overlay


@dataclass(frozen=True)
class LandingEstimate:
    frame_index: int
    timestamp_sec: float
    pixel_x: float
    pixel_y: float
    court_x_m: float | None
    court_y_m: float | None
    method: str


@dataclass(frozen=True)
class CourtRegion:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float


def load_projected_track_with_pixels(projected_track_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(projected_track_csv)
    required = {"frame_index", "timestamp_sec", "smooth_x", "smooth_y", "raw_x", "raw_y"}
    missing = required - set(df.columns)
    if missing:
        missing_joined = ", ".join(sorted(missing))
        raise ValueError(f"Track CSV missing required columns: {missing_joined}")
    output = df.copy()
    output["track_x_px"] = output["smooth_x"].where(~output["smooth_x"].isna(), output["raw_x"])
    output["track_y_px"] = output["smooth_y"].where(~output["smooth_y"].isna(), output["raw_y"])
    return output


def fit_trajectory_arc(
    track_df: pd.DataFrame,
    degree: int = 2,
    min_points: int = 8,
    landing_frame_index: int | None = None,
) -> pd.DataFrame:
    """Fit polynomial arc x(t), y(t) over visible track points."""
    valid = track_df.dropna(subset=["track_x_px", "track_y_px"]).copy()
    if landing_frame_index is not None:
        valid = valid[valid["frame_index"] <= landing_frame_index]

    output = track_df.copy()
    output["arc_x_px"] = np.nan
    output["arc_y_px"] = np.nan
    if len(valid) < min_points:
        return output

    # Normalize frame index for better numerical stability in polyfit.
    t = valid["frame_index"].to_numpy(dtype=float)
    t0 = float(t.min())
    t = t - t0
    x = valid["track_x_px"].to_numpy(dtype=float)
    y = valid["track_y_px"].to_numpy(dtype=float)
    fit_degree = min(degree, len(valid) - 1)
    if fit_degree < 1:
        return output

    x_poly = np.poly1d(np.polyfit(t, x, fit_degree))
    y_poly = np.poly1d(np.polyfit(t, y, fit_degree))
    all_t = output["frame_index"].to_numpy(dtype=float) - t0
    output["arc_x_px"] = x_poly(all_t)
    output["arc_y_px"] = y_poly(all_t)
    if landing_frame_index is not None:
        after_landing_mask = output["frame_index"] > landing_frame_index
        output.loc[after_landing_mask, ["arc_x_px", "arc_y_px"]] = np.nan
    return output


def _compute_court_speed(track_df: pd.DataFrame) -> pd.Series:
    if "court_x_m" not in track_df.columns or "court_y_m" not in track_df.columns:
        return pd.Series([np.nan] * len(track_df), index=track_df.index, dtype=float)
    dx = track_df["court_x_m"].diff()
    dy = track_df["court_y_m"].diff()
    dt = track_df["timestamp_sec"].diff().replace(0, np.nan)
    return (np.sqrt(dx * dx + dy * dy) / dt).replace([np.inf, -np.inf], np.nan)


def _is_in_region(row: pd.Series, region: CourtRegion) -> bool:
    cx = row.get("court_x_m")
    cy = row.get("court_y_m")
    if pd.isna(cx) or pd.isna(cy):
        return False
    return region.x_min_m <= float(cx) <= region.x_max_m and region.y_min_m <= float(cy) <= region.y_max_m


def detect_landing_point(
    track_df: pd.DataFrame,
    target_region: CourtRegion | None = None,
) -> LandingEstimate | None:
    """Heuristic landing detection using stop behavior + lowest near end + court region."""
    valid = track_df.dropna(subset=["track_x_px", "track_y_px"])
    if valid.empty:
        return None

    if target_region is None:
        # Default receiver-side region in standard indoor 18m x 9m court coords.
        target_region = CourtRegion(x_min_m=9.0, x_max_m=18.0, y_min_m=0.0, y_max_m=9.0)

    speeds = _compute_court_speed(track_df)
    candidates: dict[int, set[str]] = {}

    def add_candidate(idx: int, method: str) -> None:
        candidates.setdefault(idx, set()).add(method)

    # 1) Lowest point near the final third of the clip (image y increases downward).
    tail_start_frame = float(valid["frame_index"].quantile(0.66))
    tail = valid[valid["frame_index"] >= tail_start_frame]
    if not tail.empty:
        idx_lowest = int(tail["track_y_px"].idxmax())
        add_candidate(idx_lowest, "lowest_point_near_end")

    # 2) Bounce-like event: local low in trajectory after downward motion.
    y = track_df["track_y_px"]
    dy = y.diff()
    bounce_mask = (dy.shift(1) > 0.0) & (dy <= 0.0)
    bounce_indices = track_df.index[bounce_mask.fillna(False)]
    if len(bounce_indices) > 0:
        add_candidate(int(bounce_indices[0]), "bounce_behavior")

    # 3) Sudden stop: speed drops sharply after being high.
    if speeds.notna().sum() >= 4:
        rolling_prev = speeds.rolling(window=3, min_periods=2).mean().shift(1)
        stop_mask = (rolling_prev > 4.0) & (speeds < 1.5)
        stop_indices = track_df.index[stop_mask.fillna(False)]
        if len(stop_indices) > 0:
            add_candidate(int(stop_indices[0]), "sudden_stop")

    # 4) First contact with target region in calibrated coordinates.
    if {"court_x_m", "court_y_m"}.issubset(track_df.columns):
        in_target = track_df.apply(lambda row: _is_in_region(row, target_region), axis=1)
        target_indices = track_df.index[in_target]
        if len(target_indices) > 0:
            add_candidate(int(target_indices[0]), "target_region_contact")

    # 5) Last tracked point inside nominal court bounds.
    if {"court_x_m", "court_y_m"}.issubset(track_df.columns):
        in_court = track_df[
            track_df["court_x_m"].between(0.0, 18.0, inclusive="both")
            & track_df["court_y_m"].between(0.0, 9.0, inclusive="both")
        ]
        in_court = in_court.dropna(subset=["track_x_px", "track_y_px"])
        if not in_court.empty:
            idx_last_court = int(in_court.index[-1])
            add_candidate(idx_last_court, "last_in_court")

    if not candidates:
        final_idx = int(valid.index[-1])
        row = track_df.loc[final_idx]
        return LandingEstimate(
            frame_index=int(row["frame_index"]),
            timestamp_sec=float(row["timestamp_sec"]),
            pixel_x=float(row["track_x_px"]),
            pixel_y=float(row["track_y_px"]),
            court_x_m=float(row["court_x_m"]) if pd.notna(row.get("court_x_m")) else None,
            court_y_m=float(row["court_y_m"]) if pd.notna(row.get("court_y_m")) else None,
            method="last_valid_point_fallback",
        )

    method_weights = {
        "target_region_contact": 4.0,
        "bounce_behavior": 2.0,
        "sudden_stop": 2.0,
        "lowest_point_near_end": 1.0,
        "last_in_court": 1.0,
    }

    # Prefer higher confidence evidence; break ties with later frames.
    ranked = sorted(
        candidates.items(),
        key=lambda item: (sum(method_weights.get(m, 0.5) for m in item[1]), item[0]),
        reverse=True,
    )
    chosen_idx = int(ranked[0][0])
    chosen_row = track_df.loc[chosen_idx]
    methods = ",".join(sorted(candidates[chosen_idx]))
    return LandingEstimate(
        frame_index=int(chosen_row["frame_index"]),
        timestamp_sec=float(chosen_row["timestamp_sec"]),
        pixel_x=float(chosen_row["track_x_px"]),
        pixel_y=float(chosen_row["track_y_px"]),
        court_x_m=float(chosen_row["court_x_m"]) if pd.notna(chosen_row.get("court_x_m")) else None,
        court_y_m=float(chosen_row["court_y_m"]) if pd.notna(chosen_row.get("court_y_m")) else None,
        method=methods,
    )


def export_arc_csv(track_with_arc_df: pd.DataFrame, out_csv: str | Path) -> None:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    track_with_arc_df.to_csv(path, index=False)


def export_landing_json(landing: LandingEstimate | None, out_json: str | Path) -> None:
    payload: dict[str, object] = {
        "assumption": "estimated landing from calibrated 2d trajectory behavior",
        "landing": None if landing is None else asdict(landing),
    }
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_arc_overlay_video(
    input_video: str | Path,
    track_with_arc_df: pd.DataFrame,
    landing: LandingEstimate | None,
    output_video: str | Path,
) -> None:
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    df = track_with_arc_df.copy()
    frame_to_arc: dict[int, tuple[float, float]] = {}
    frame_to_track: dict[int, tuple[float, float]] = {}
    for row in df.itertuples(index=False):
        frame_idx = int(getattr(row, "frame_index"))
        arc_x = getattr(row, "arc_x_px")
        arc_y = getattr(row, "arc_y_px")
        track_x = getattr(row, "track_x_px")
        track_y = getattr(row, "track_y_px")
        if pd.notna(arc_x) and pd.notna(arc_y):
            frame_to_arc[frame_idx] = (float(arc_x), float(arc_y))
        if pd.notna(track_x) and pd.notna(track_y):
            frame_to_track[frame_idx] = (float(track_x), float(track_y))

    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Draw fitted arc up to current frame.
        arc_points: list[tuple[int, int]] = []
        for i in range(max(0, frame_idx - 120), frame_idx + 1):
            p = frame_to_arc.get(i)
            if p is not None:
                arc_points.append((int(p[0]), int(p[1])))
        if len(arc_points) >= 2:
            for a, b in zip(arc_points[:-1], arc_points[1:]):
                # Glow + core stroke for a cleaner highlight effect.
                cv2.line(frame, a, b, (20, 120, 255), 5, cv2.LINE_AA)
                cv2.line(frame, a, b, (120, 210, 255), 2, cv2.LINE_AA)

        # Draw measured current point.
        current = frame_to_track.get(frame_idx)
        if current is not None:
            cx, cy = int(current[0]), int(current[1])
            cv2.circle(frame, (cx, cy), 8, (0, 60, 0), -1)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 120), -1)

        if landing is not None and frame_idx >= landing.frame_index:
            lx, ly = int(landing.pixel_x), int(landing.pixel_y)
            cv2.circle(frame, (lx, ly), 10, (10, 10, 180), -1)
            cv2.circle(frame, (lx, ly), 5, (20, 20, 255), -1)
            cv2.line(frame, (lx - 16, ly), (lx + 16, ly), (40, 40, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (lx, ly - 16), (lx, ly + 16), (40, 40, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                "Estimated landing",
                (lx + 12, ly - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "Trajectory Arc (fit) + landing estimate",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()


def render_court_landing_map(landing: LandingEstimate | None, out_image: str | Path) -> None:
    if landing is None or landing.court_x_m is None or landing.court_y_m is None:
        draw_topdown_court_overlay(landing_point_m=None, out_image=out_image)
        return
    draw_topdown_court_overlay(
        landing_point_m=(landing.court_x_m, landing.court_y_m),
        out_image=out_image,
    )


def run_trajectory_analysis(
    projected_track_csv: str | Path,
    degree: int = 2,
) -> tuple[pd.DataFrame, LandingEstimate | None]:
    track_df = load_projected_track_with_pixels(projected_track_csv)
    landing = detect_landing_point(track_df)
    arc_df = fit_trajectory_arc(
        track_df,
        degree=degree,
        landing_frame_index=None if landing is None else landing.frame_index,
    )
    if landing is None:
        landing = detect_landing_point(arc_df)
    return arc_df, landing
