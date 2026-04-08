from src.detection.ball_detector import BallDetection
from src.tracking.single_ball_tracker import (
    TrackPoint,
    interpolate_missing,
    select_detection,
    smooth_track,
)


def test_select_detection_uses_highest_conf_without_previous() -> None:
    detections = [
        BallDetection(0, 0, 10, 10, confidence=0.4, class_id=0),
        BallDetection(0, 0, 10, 10, confidence=0.9, class_id=0),
    ]
    chosen = select_detection(detections, previous_xy=None, max_step_px=100.0)
    assert chosen is not None
    assert chosen.confidence == 0.9


def test_select_detection_uses_nearest_neighbor_with_previous() -> None:
    previous_xy = (100.0, 100.0)
    near = BallDetection(96, 96, 104, 104, confidence=0.5, class_id=0)
    far = BallDetection(250, 250, 260, 260, confidence=0.99, class_id=0)
    chosen = select_detection([far, near], previous_xy=previous_xy, max_step_px=50.0)
    assert chosen is near


def test_select_detection_rejects_large_jump() -> None:
    previous_xy = (50.0, 50.0)
    far = BallDetection(300, 300, 320, 320, confidence=0.99, class_id=0)
    chosen = select_detection([far], previous_xy=previous_xy, max_step_px=100.0)
    assert chosen is None


def test_interpolate_missing_fills_short_gap() -> None:
    points = [
        TrackPoint(0, 0.0, True, False, 0.0, 0.0),
        TrackPoint(1, 0.1, False, False, None, None),
        TrackPoint(2, 0.2, False, False, None, None),
        TrackPoint(3, 0.3, True, False, 30.0, 30.0),
    ]
    interpolate_missing(points, max_gap=2)
    assert points[1].raw_x == 10.0
    assert points[1].raw_y == 10.0
    assert points[2].raw_x == 20.0
    assert points[2].raw_y == 20.0
    assert points[1].interpolated
    assert points[2].interpolated


def test_smooth_track_applies_moving_average() -> None:
    points = [
        TrackPoint(0, 0.0, True, False, 0.0, 0.0),
        TrackPoint(1, 0.1, True, False, 10.0, 10.0),
        TrackPoint(2, 0.2, True, False, 20.0, 20.0),
    ]
    smooth_track(points, window_size=3)
    assert points[1].smooth_x == 10.0
    assert points[1].smooth_y == 10.0
    assert points[0].smooth_x == 5.0
    assert points[2].smooth_x == 15.0
