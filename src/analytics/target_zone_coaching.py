"""Target-zone coaching analytics for Sprint 8."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.trajectory import CourtRegion


@dataclass(frozen=True)
class TargetCoachingSummary:
    serve_count: int
    scored_serves: int
    avg_distance_to_target_m: float | None
    best_distance_to_target_m: float | None
    bullseye_rate_pct: float | None
    average_target_proximity_score: float | None
    coaching_note: str


def target_center(region: CourtRegion) -> tuple[float, float]:
    return ((region.x_min_m + region.x_max_m) / 2.0, (region.y_min_m + region.y_max_m) / 2.0)


def distance_to_target_center(
    landing_x_m: float | None,
    landing_y_m: float | None,
    target_region: CourtRegion,
) -> float | None:
    if landing_x_m is None or landing_y_m is None:
        return None
    tx, ty = target_center(target_region)
    return float(np.hypot(landing_x_m - tx, landing_y_m - ty))


def target_proximity_score(distance_m: float | None, reference_radius_m: float = 3.0) -> float | None:
    if distance_m is None:
        return None
    score = 100.0 * max(0.0, 1.0 - (distance_m / reference_radius_m))
    return float(np.clip(score, 0.0, 100.0))


def score_serves_for_target_zone(
    serves_df: pd.DataFrame,
    target_region: CourtRegion,
    bullseye_radius_m: float = 0.75,
) -> pd.DataFrame:
    scored = serves_df.copy()

    distances: list[float | None] = []
    scores: list[float | None] = []
    bullseyes: list[bool | None] = []

    for row in scored.itertuples(index=False):
        distance = distance_to_target_center(
            landing_x_m=getattr(row, "landing_x_m", None),
            landing_y_m=getattr(row, "landing_y_m", None),
            target_region=target_region,
        )
        score = target_proximity_score(distance)
        bullseye = None if distance is None else distance <= bullseye_radius_m

        distances.append(distance)
        scores.append(score)
        bullseyes.append(bullseye)

    scored["distance_to_target_m"] = distances
    scored["target_proximity_score"] = scores
    scored["bullseye_hit"] = bullseyes
    return scored


def summarize_target_coaching(scored_df: pd.DataFrame) -> TargetCoachingSummary:
    distance_values = np.array(
        [value for value in scored_df.get("distance_to_target_m", pd.Series(dtype=float)).tolist() if pd.notna(value)],
        dtype=float,
    )
    score_values = np.array(
        [value for value in scored_df.get("target_proximity_score", pd.Series(dtype=float)).tolist() if pd.notna(value)],
        dtype=float,
    )
    bullseye_values = np.array(
        [1.0 if value else 0.0 for value in scored_df.get("bullseye_hit", pd.Series(dtype=bool)).tolist() if pd.notna(value)],
        dtype=float,
    )

    avg_distance = None if distance_values.size == 0 else float(np.mean(distance_values))
    best_distance = None if distance_values.size == 0 else float(np.min(distance_values))
    avg_score = None if score_values.size == 0 else float(np.mean(score_values))
    bullseye_rate = None if bullseye_values.size == 0 else float(np.mean(bullseye_values) * 100.0)

    note: str
    if avg_score is None:
        note = "No valid landings yet. Run calibration and trajectory on more serves."
    elif avg_score >= 80.0:
        note = "Excellent target control. Keep the same toss/contact rhythm."
    elif avg_score >= 60.0:
        note = "Solid control. Focus on reducing lateral drift for tighter grouping."
    else:
        note = "Target control needs work. Prioritize repeatable toss height and contact timing."

    return TargetCoachingSummary(
        serve_count=len(scored_df),
        scored_serves=int(distance_values.size),
        avg_distance_to_target_m=avg_distance,
        best_distance_to_target_m=best_distance,
        bullseye_rate_pct=bullseye_rate,
        average_target_proximity_score=avg_score,
        coaching_note=note,
    )
