from pathlib import Path

import pandas as pd

from src.analytics.trajectory import (
    CourtRegion,
    detect_landing_point,
    fit_trajectory_arc,
    load_projected_track_with_pixels,
    run_trajectory_analysis,
)


def test_fit_trajectory_arc_returns_fitted_columns() -> None:
    rows: list[dict[str, float]] = []
    for frame in range(20):
        x = float(frame * 5)
        y = float(0.2 * (frame - 10) ** 2 + 50)
        rows.append(
            {
                "frame_index": frame,
                "timestamp_sec": frame / 60.0,
                "smooth_x": x,
                "smooth_y": y,
                "raw_x": x,
                "raw_y": y,
                "court_x_m": frame * 0.1,
                "court_y_m": 4.0,
            }
        )
    df = pd.DataFrame(rows)
    df["track_x_px"] = df["smooth_x"]
    df["track_y_px"] = df["smooth_y"]
    out = fit_trajectory_arc(df, degree=2)
    assert "arc_x_px" in out.columns
    assert "arc_y_px" in out.columns
    assert out["arc_x_px"].notna().sum() == len(out)
    assert out["arc_y_px"].notna().sum() == len(out)


def test_detect_landing_point_uses_tail_lowest() -> None:
    df = pd.DataFrame(
        [
            {"frame_index": 0, "timestamp_sec": 0.0, "track_x_px": 100.0, "track_y_px": 100.0, "court_x_m": 1.0, "court_y_m": 1.0},
            {"frame_index": 1, "timestamp_sec": 0.1, "track_x_px": 120.0, "track_y_px": 110.0, "court_x_m": 2.0, "court_y_m": 1.0},
            {"frame_index": 2, "timestamp_sec": 0.2, "track_x_px": 140.0, "track_y_px": 140.0, "court_x_m": 3.0, "court_y_m": 1.0},
            {"frame_index": 3, "timestamp_sec": 0.3, "track_x_px": 150.0, "track_y_px": 170.0, "court_x_m": 3.2, "court_y_m": 1.2},
        ]
    )
    landing = detect_landing_point(df)
    assert landing is not None
    assert landing.frame_index in (2, 3)
    assert landing.method != ""


def test_run_trajectory_analysis_pipeline(tmp_path: Path) -> None:
    csv_path = tmp_path / "projected.csv"
    pd.DataFrame(
        [
            {"frame_index": 0, "timestamp_sec": 0.0, "smooth_x": 10.0, "smooth_y": 20.0, "raw_x": 10.0, "raw_y": 20.0, "court_x_m": 0.5, "court_y_m": 2.0},
            {"frame_index": 1, "timestamp_sec": 0.1, "smooth_x": 20.0, "smooth_y": 25.0, "raw_x": 20.0, "raw_y": 25.0, "court_x_m": 0.8, "court_y_m": 2.2},
            {"frame_index": 2, "timestamp_sec": 0.2, "smooth_x": 30.0, "smooth_y": 35.0, "raw_x": 30.0, "raw_y": 35.0, "court_x_m": 1.0, "court_y_m": 2.6},
            {"frame_index": 3, "timestamp_sec": 0.3, "smooth_x": 40.0, "smooth_y": 50.0, "raw_x": 40.0, "raw_y": 50.0, "court_x_m": 1.1, "court_y_m": 3.0},
            {"frame_index": 4, "timestamp_sec": 0.4, "smooth_x": 50.0, "smooth_y": 65.0, "raw_x": 50.0, "raw_y": 65.0, "court_x_m": 1.2, "court_y_m": 3.4},
            {"frame_index": 5, "timestamp_sec": 0.5, "smooth_x": 60.0, "smooth_y": 80.0, "raw_x": 60.0, "raw_y": 80.0, "court_x_m": 1.25, "court_y_m": 3.7},
            {"frame_index": 6, "timestamp_sec": 0.6, "smooth_x": 70.0, "smooth_y": 90.0, "raw_x": 70.0, "raw_y": 90.0, "court_x_m": 1.3, "court_y_m": 4.0},
            {"frame_index": 7, "timestamp_sec": 0.7, "smooth_x": 80.0, "smooth_y": 98.0, "raw_x": 80.0, "raw_y": 98.0, "court_x_m": 1.35, "court_y_m": 4.2},
        ]
    ).to_csv(csv_path, index=False)
    arc_df, landing = run_trajectory_analysis(csv_path, degree=2)
    assert "arc_x_px" in arc_df.columns
    assert "arc_y_px" in arc_df.columns
    assert landing is not None


def test_load_projected_track_with_pixels_uses_fallback(tmp_path: Path) -> None:
    csv_path = tmp_path / "projected.csv"
    pd.DataFrame(
        [
            {"frame_index": 0, "timestamp_sec": 0.0, "smooth_x": None, "smooth_y": None, "raw_x": 12.0, "raw_y": 22.0},
            {"frame_index": 1, "timestamp_sec": 0.1, "smooth_x": 15.0, "smooth_y": 25.0, "raw_x": 14.0, "raw_y": 24.0},
        ]
    ).to_csv(csv_path, index=False)
    out = load_projected_track_with_pixels(csv_path)
    assert out["track_x_px"].tolist() == [12.0, 15.0]
    assert out["track_y_px"].tolist() == [22.0, 25.0]


def test_detect_landing_prefers_target_region_contact() -> None:
    df = pd.DataFrame(
        [
            {"frame_index": 0, "timestamp_sec": 0.0, "track_x_px": 50.0, "track_y_px": 20.0, "court_x_m": 7.0, "court_y_m": 4.0},
            {"frame_index": 1, "timestamp_sec": 0.1, "track_x_px": 60.0, "track_y_px": 30.0, "court_x_m": 8.5, "court_y_m": 4.0},
            {"frame_index": 2, "timestamp_sec": 0.2, "track_x_px": 70.0, "track_y_px": 45.0, "court_x_m": 9.2, "court_y_m": 4.2},
            {"frame_index": 3, "timestamp_sec": 0.3, "track_x_px": 72.0, "track_y_px": 44.0, "court_x_m": 10.0, "court_y_m": 4.3},
        ]
    )
    landing = detect_landing_point(df, target_region=CourtRegion(9.0, 18.0, 0.0, 9.0))
    assert landing is not None
    assert landing.frame_index == 2
    assert "target_region_contact" in landing.method


def test_fit_trajectory_arc_masks_points_after_landing() -> None:
    df = pd.DataFrame(
        [
            {"frame_index": 0, "timestamp_sec": 0.0, "track_x_px": 10.0, "track_y_px": 10.0},
            {"frame_index": 1, "timestamp_sec": 0.1, "track_x_px": 20.0, "track_y_px": 20.0},
            {"frame_index": 2, "timestamp_sec": 0.2, "track_x_px": 30.0, "track_y_px": 35.0},
            {"frame_index": 3, "timestamp_sec": 0.3, "track_x_px": 40.0, "track_y_px": 50.0},
            {"frame_index": 4, "timestamp_sec": 0.4, "track_x_px": 50.0, "track_y_px": 60.0},
            {"frame_index": 5, "timestamp_sec": 0.5, "track_x_px": 60.0, "track_y_px": 68.0},
            {"frame_index": 6, "timestamp_sec": 0.6, "track_x_px": 70.0, "track_y_px": 74.0},
            {"frame_index": 7, "timestamp_sec": 0.7, "track_x_px": 80.0, "track_y_px": 78.0},
            {"frame_index": 8, "timestamp_sec": 0.8, "track_x_px": 90.0, "track_y_px": 81.0},
        ]
    )
    out = fit_trajectory_arc(df, degree=2, min_points=6, landing_frame_index=5)
    assert out.loc[out["frame_index"] <= 5, "arc_x_px"].notna().all()
    assert out.loc[out["frame_index"] > 5, "arc_x_px"].isna().all()
