from pathlib import Path

import pandas as pd

from src.analytics.session_consistency import compute_session_from_manifest
from src.analytics.trajectory import CourtRegion


def _write_track_csv(path: Path, landing_x: float, landing_y: float, x_scale: float = 1.0) -> None:
    rows: list[dict[str, float]] = []
    for i in range(10):
        progress = i / 9.0
        court_x = x_scale * (landing_x * progress)
        court_y = landing_y * progress
        rows.append(
            {
                "frame_index": i,
                "timestamp_sec": i * 0.02,
                "smooth_x": 100.0 + i * 4.0,
                "smooth_y": 120.0 + i * 6.0,
                "raw_x": 100.0 + i * 4.0,
                "raw_y": 120.0 + i * 6.0,
                "court_x_m": court_x,
                "court_y_m": court_y,
            }
        )

    # Ensure final row is the intended landing point.
    rows[-1]["court_x_m"] = landing_x
    rows[-1]["court_y_m"] = landing_y

    pd.DataFrame(rows).to_csv(path, index=False)


def test_compute_session_from_manifest_outputs_summary(tmp_path: Path) -> None:
    track1 = tmp_path / "serve_01.csv"
    track2 = tmp_path / "serve_02.csv"
    track3 = tmp_path / "serve_03.csv"

    _write_track_csv(track1, landing_x=11.0, landing_y=3.0, x_scale=1.0)
    _write_track_csv(track2, landing_x=12.0, landing_y=4.0, x_scale=1.1)
    _write_track_csv(track3, landing_x=19.0, landing_y=10.0, x_scale=1.3)

    manifest = tmp_path / "session.csv"
    pd.DataFrame(
        [
            {"serve_id": "s1", "projected_track_csv": "serve_01.csv"},
            {"serve_id": "s2", "projected_track_csv": "serve_02.csv"},
            {"serve_id": "s3", "projected_track_csv": "serve_03.csv"},
        ]
    ).to_csv(manifest, index=False)

    target_region = CourtRegion(10.5, 12.5, 2.5, 4.5)
    serves_df, summary = compute_session_from_manifest(manifest, target_region=target_region)

    assert len(serves_df) == 3
    assert summary.serve_count == 3
    assert summary.average_speed_kph is not None
    assert summary.speed_variance_kph2 is not None
    assert summary.in_percentage is not None
    assert summary.out_percentage is not None
    assert summary.target_zone_accuracy is not None
    assert summary.landing_zone_spread_m is not None
    assert summary.depth_consistency_m is not None
    assert summary.consistency_score is not None


def test_compute_session_in_out_and_target_counts(tmp_path: Path) -> None:
    track1 = tmp_path / "serve_a.csv"
    track2 = tmp_path / "serve_b.csv"

    _write_track_csv(track1, landing_x=11.2, landing_y=3.5)
    _write_track_csv(track2, landing_x=18.4, landing_y=9.6)

    manifest = tmp_path / "session.csv"
    pd.DataFrame(
        [
            {"serve_id": "a", "projected_track_csv": "serve_a.csv"},
            {"serve_id": "b", "projected_track_csv": "serve_b.csv"},
        ]
    ).to_csv(manifest, index=False)

    target_region = CourtRegion(11.0, 12.0, 3.0, 4.0)
    serves_df, summary = compute_session_from_manifest(manifest, target_region=target_region)

    assert serves_df["target_hit"].tolist() == [True, False]
    assert serves_df["in_bounds"].tolist() == [True, False]
    assert summary.in_percentage == 50.0
    assert summary.out_percentage == 50.0
    assert summary.target_zone_accuracy == 50.0
