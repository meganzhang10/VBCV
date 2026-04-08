"""Session-level serve consistency analytics for Sprint 6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.speed_estimation import run_speed_estimation
from src.analytics.trajectory import CourtRegion, detect_landing_point, load_projected_track_with_pixels


@dataclass(frozen=True)
class ServeSessionEntry:
    serve_id: str
    projected_track_csv: str


@dataclass(frozen=True)
class ServeSummary:
    serve_id: str
    projected_track_csv: str
    speed_kph: float | None
    landing_x_m: float | None
    landing_y_m: float | None
    in_bounds: bool | None
    target_hit: bool | None
    landing_method: str | None


@dataclass(frozen=True)
class SessionSummary:
    serve_count: int
    average_speed_kph: float | None
    speed_variance_kph2: float | None
    landing_zone_spread_m: float | None
    in_percentage: float | None
    out_percentage: float | None
    target_zone_accuracy: float | None
    left_right_bias_m: float | None
    depth_consistency_m: float | None
    consistency_score: float | None
    assumption: str


def _resolve_path(base_dir: Path, raw_path: str) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_session_manifest(manifest_csv: str | Path) -> list[ServeSessionEntry]:
    path = Path(manifest_csv)
    df = pd.read_csv(path)
    required = {"serve_id", "projected_track_csv"}
    missing = required - set(df.columns)
    if missing:
        missing_joined = ", ".join(sorted(missing))
        raise ValueError(f"Session manifest missing required columns: {missing_joined}")

    base_dir = path.parent
    entries: list[ServeSessionEntry] = []
    for row in df.itertuples(index=False):
        entries.append(
            ServeSessionEntry(
                serve_id=str(getattr(row, "serve_id")),
                projected_track_csv=_resolve_path(base_dir, str(getattr(row, "projected_track_csv"))),
            )
        )
    return entries


def _is_inside(region: CourtRegion, x: float | None, y: float | None) -> bool | None:
    if x is None or y is None:
        return None
    return region.x_min_m <= x <= region.x_max_m and region.y_min_m <= y <= region.y_max_m


def compute_serve_summary(
    entry: ServeSessionEntry,
    target_region: CourtRegion,
    court_bounds: CourtRegion,
    early_window_sec: float = 0.3,
) -> ServeSummary:
    _, speed = run_speed_estimation(entry.projected_track_csv, early_window_sec=early_window_sec)

    track_df = load_projected_track_with_pixels(entry.projected_track_csv)
    landing = detect_landing_point(track_df, target_region=target_region)

    landing_x = None if landing is None else landing.court_x_m
    landing_y = None if landing is None else landing.court_y_m
    in_bounds = _is_inside(court_bounds, landing_x, landing_y)
    target_hit = _is_inside(target_region, landing_x, landing_y)

    return ServeSummary(
        serve_id=entry.serve_id,
        projected_track_csv=entry.projected_track_csv,
        speed_kph=speed.max_speed_kph,
        landing_x_m=landing_x,
        landing_y_m=landing_y,
        in_bounds=in_bounds,
        target_hit=target_hit,
        landing_method=None if landing is None else landing.method,
    )


def _mean_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _var_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.var(values))


def _std_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.std(values))


def _consistency_score(
    speed_std: float | None,
    spread: float | None,
    in_percentage: float | None,
    target_accuracy: float | None,
    depth_std: float | None,
) -> float | None:
    components: list[tuple[float, float]] = []

    if speed_std is not None:
        speed_consistency = 100.0 / (1.0 + (speed_std / 15.0))
        components.append((speed_consistency, 0.30))

    if spread is not None:
        landing_consistency = 100.0 / (1.0 + (spread / 2.5))
        components.append((landing_consistency, 0.25))

    if in_percentage is not None:
        components.append((in_percentage, 0.20))

    if target_accuracy is not None:
        components.append((target_accuracy, 0.15))

    if depth_std is not None:
        depth_score = 100.0 / (1.0 + (depth_std / 2.0))
        components.append((depth_score, 0.10))

    if not components:
        return None

    weighted_sum = sum(value * weight for value, weight in components)
    weight_total = sum(weight for _, weight in components)
    return float(np.clip(weighted_sum / weight_total, 0.0, 100.0))


def summarize_session(serves: list[ServeSummary]) -> SessionSummary:
    speeds = np.array([s.speed_kph for s in serves if s.speed_kph is not None], dtype=float)
    landings = np.array(
        [[s.landing_x_m, s.landing_y_m] for s in serves if s.landing_x_m is not None and s.landing_y_m is not None],
        dtype=float,
    )

    average_speed = _mean_or_none(speeds)
    speed_variance = _var_or_none(speeds)

    spread = None
    left_right_bias = None
    depth_consistency = None
    if landings.size > 0:
        x_values = landings[:, 0]
        y_values = landings[:, 1]
        spread = float(np.sqrt(np.var(x_values) + np.var(y_values)))
        left_right_bias = float(np.mean(y_values - 4.5))
        depth_consistency = _std_or_none(x_values)

    in_values = np.array([1.0 if s.in_bounds else 0.0 for s in serves if s.in_bounds is not None], dtype=float)
    target_values = np.array(
        [1.0 if s.target_hit else 0.0 for s in serves if s.target_hit is not None],
        dtype=float,
    )

    in_percentage = None if in_values.size == 0 else float(np.mean(in_values) * 100.0)
    out_percentage = None if in_percentage is None else float(100.0 - in_percentage)
    target_accuracy = None if target_values.size == 0 else float(np.mean(target_values) * 100.0)

    speed_std = _std_or_none(speeds)
    score = _consistency_score(speed_std, spread, in_percentage, target_accuracy, depth_consistency)

    return SessionSummary(
        serve_count=len(serves),
        average_speed_kph=average_speed,
        speed_variance_kph2=speed_variance,
        landing_zone_spread_m=spread,
        in_percentage=in_percentage,
        out_percentage=out_percentage,
        target_zone_accuracy=target_accuracy,
        left_right_bias_m=left_right_bias,
        depth_consistency_m=depth_consistency,
        consistency_score=score,
        assumption=(
            "session metrics estimated from calibrated 2d tracks; speed is 2d court-plane speed and "
            "landing is heuristic"
        ),
    )


def compute_session_from_manifest(
    manifest_csv: str | Path,
    target_region: CourtRegion,
    court_bounds: CourtRegion | None = None,
    early_window_sec: float = 0.3,
) -> tuple[pd.DataFrame, SessionSummary]:
    entries = load_session_manifest(manifest_csv)
    bounds = court_bounds or CourtRegion(0.0, 18.0, 0.0, 9.0)

    serves = [
        compute_serve_summary(
            entry,
            target_region=target_region,
            court_bounds=bounds,
            early_window_sec=early_window_sec,
        )
        for entry in entries
    ]
    serves_df = pd.DataFrame(asdict(s) for s in serves)
    return serves_df, summarize_session(serves)


def export_session_summary_json(summary: SessionSummary, out_json: str | Path) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
