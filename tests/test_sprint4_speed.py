from pathlib import Path

import pandas as pd

from src.analytics.speed_estimation import (
    compute_instantaneous_speed,
    run_speed_estimation,
    smooth_court_track,
    summarize_serve_speed,
)


def test_compute_instantaneous_speed_linear_motion() -> None:
    df = pd.DataFrame(
        [
            {"frame_index": 0, "timestamp_sec": 0.0, "court_x_smooth_m": 0.0, "court_y_smooth_m": 0.0},
            {"frame_index": 1, "timestamp_sec": 0.1, "court_x_smooth_m": 1.0, "court_y_smooth_m": 0.0},
            {"frame_index": 2, "timestamp_sec": 0.2, "court_x_smooth_m": 2.0, "court_y_smooth_m": 0.0},
        ]
    )
    out = compute_instantaneous_speed(df)
    assert pd.isna(out.iloc[0]["inst_speed_mps"])
    assert abs(float(out.iloc[1]["inst_speed_mps"]) - 10.0) < 1e-6
    assert abs(float(out.iloc[2]["inst_speed_mps"]) - 10.0) < 1e-6


def test_summarize_serve_speed_first_0_3s_window() -> None:
    df = pd.DataFrame(
        [
            {"timestamp_sec": 0.00, "inst_speed_mps": 8.0},
            {"timestamp_sec": 0.10, "inst_speed_mps": 10.0},
            {"timestamp_sec": 0.20, "inst_speed_mps": 12.0},
            {"timestamp_sec": 0.50, "inst_speed_mps": 20.0},
        ]
    )
    summary = summarize_serve_speed(df, early_window_sec=0.3)
    assert summary.max_speed_mps == 20.0
    assert summary.avg_speed_first_0_3s_mps == 10.0
    assert summary.assumption == "estimated serve speed from calibrated 2d motion"


def test_run_speed_estimation_pipeline(tmp_path: Path) -> None:
    track_csv = tmp_path / "projected.csv"
    pd.DataFrame(
        [
            {"frame_index": 0, "timestamp_sec": 0.0, "court_x_m": 0.0, "court_y_m": 0.0},
            {"frame_index": 1, "timestamp_sec": 0.1, "court_x_m": 1.0, "court_y_m": 0.0},
            {"frame_index": 2, "timestamp_sec": 0.2, "court_x_m": 2.0, "court_y_m": 0.0},
            {"frame_index": 3, "timestamp_sec": 0.3, "court_x_m": 3.0, "court_y_m": 0.0},
        ]
    ).to_csv(track_csv, index=False)
    speed_df, summary = run_speed_estimation(track_csv, smooth_window=1, early_window_sec=0.3)
    assert "inst_speed_mps" in speed_df.columns
    assert summary.max_speed_mps is not None
    assert summary.max_speed_mps > 0


def test_smooth_court_track_window_one_keeps_values() -> None:
    df = pd.DataFrame(
        [
            {"court_x_m": 1.0, "court_y_m": 2.0},
            {"court_x_m": 3.0, "court_y_m": 4.0},
        ]
    )
    out = smooth_court_track(df, window_size=1)
    assert out["court_x_smooth_m"].tolist() == [1.0, 3.0]
    assert out["court_y_smooth_m"].tolist() == [2.0, 4.0]
