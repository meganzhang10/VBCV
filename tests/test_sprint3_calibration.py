from pathlib import Path
import json

import pandas as pd

from src.analytics.court_calibration import (
    calibrate_and_project,
    estimate_homography,
    estimate_landing_point,
    estimate_speed_mps,
    load_point_pairs,
)


def test_estimate_homography_maps_points() -> None:
    # Synthetic mapping: court_x = pixel_x / 100, court_y = pixel_y / 100
    points_json = {
        "points": [
            {"name": "p0", "pixel": [0, 0], "court": [0, 0]},
            {"name": "p1", "pixel": [100, 0], "court": [1, 0]},
            {"name": "p2", "pixel": [0, 100], "court": [0, 1]},
            {"name": "p3", "pixel": [200, 100], "court": [2, 1]},
        ]
    }
    points = [
        load_point_pairs_from_item(item, index=i) for i, item in enumerate(points_json["points"])
    ]
    calibration = estimate_homography(points)
    cx, cy = calibration.pixel_to_court(150, 50)
    assert abs(cx - 1.5) < 1e-3
    assert abs(cy - 0.5) < 1e-3


def load_point_pairs_from_item(item: dict[str, object], index: int):
    from src.analytics.court_calibration import PointPair

    pixel = item["pixel"]
    court = item["court"]
    assert isinstance(pixel, list)
    assert isinstance(court, list)
    return PointPair(
        name=str(item.get("name", f"p{index}")),
        pixel=(float(pixel[0]), float(pixel[1])),
        court=(float(court[0]), float(court[1])),
    )


def test_calibrate_and_project_outputs_court_coords(tmp_path: Path) -> None:
    points_path = tmp_path / "points.json"
    points_path.write_text(
        json.dumps(
            {
                "points": [
                    {"name": "a", "pixel": [0, 0], "court": [0, 0]},
                    {"name": "b", "pixel": [100, 0], "court": [1, 0]},
                    {"name": "c", "pixel": [0, 100], "court": [0, 1]},
                    {"name": "d", "pixel": [100, 100], "court": [1, 1]},
                ]
            }
        ),
        encoding="utf-8",
    )
    track_path = tmp_path / "track.csv"
    pd.DataFrame(
        [
            {"frame_index": 0, "timestamp_sec": 0.0, "raw_x": 0.0, "raw_y": 0.0, "smooth_x": 0.0, "smooth_y": 0.0},
            {"frame_index": 1, "timestamp_sec": 0.1, "raw_x": 50.0, "raw_y": 50.0, "smooth_x": 50.0, "smooth_y": 50.0},
            {"frame_index": 2, "timestamp_sec": 0.2, "raw_x": 100.0, "raw_y": 100.0, "smooth_x": 100.0, "smooth_y": 100.0},
        ]
    ).to_csv(track_path, index=False)

    _, projected, landing, speed = calibrate_and_project(points_path, track_path)
    assert "court_x_m" in projected.columns
    assert "court_y_m" in projected.columns
    assert landing is not None
    assert abs(landing[0] - 1.0) < 1e-3
    assert abs(landing[1] - 1.0) < 1e-3
    assert speed is not None
    assert speed > 0


def test_load_point_pairs_requires_four_points(tmp_path: Path) -> None:
    points_path = tmp_path / "bad_points.json"
    points_path.write_text(
        json.dumps({"points": [{"name": "a", "pixel": [0, 0], "court": [0, 0]}]}),
        encoding="utf-8",
    )
    try:
        load_point_pairs(points_path)
        assert False, "Expected ValueError for too few points"
    except ValueError:
        pass


def test_estimate_landing_and_speed_on_empty_data() -> None:
    df = pd.DataFrame(
        [
            {"timestamp_sec": 0.0, "court_x_m": None, "court_y_m": None},
            {"timestamp_sec": 0.1, "court_x_m": None, "court_y_m": None},
        ]
    )
    assert estimate_landing_point(df) is None
    assert estimate_speed_mps(df) is None
