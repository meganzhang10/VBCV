"""Serve-level metrics helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from src.analytics.speed_estimation import run_speed_estimation


def compute_serve_metrics(
    projected_track_csv: str | Path,
    early_window_sec: float = 0.3,
) -> dict[str, float | str | None]:
    """Compute primary serve metrics from calibrated 2D track."""
    _, summary = run_speed_estimation(
        projected_track_csv=projected_track_csv,
        smooth_window=5,
        early_window_sec=early_window_sec,
    )
    return asdict(summary)


def write_serve_metrics_json(
    projected_track_csv: str | Path,
    out_json: str | Path,
    early_window_sec: float = 0.3,
) -> None:
    metrics = compute_serve_metrics(projected_track_csv, early_window_sec=early_window_sec)
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
