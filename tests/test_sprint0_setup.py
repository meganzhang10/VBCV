from pathlib import Path

from src.config import INPUT_SPEC, PHASED_METRICS


def test_directory_layout_exists() -> None:
    required_dirs = [
        Path("data/raw/videos"),
        Path("data/processed"),
        Path("models"),
        Path("src/detection"),
        Path("src/tracking"),
        Path("src/analytics"),
        Path("src/visualization"),
    ]
    for directory in required_dirs:
        assert directory.exists(), f"Missing directory: {directory}"


def test_input_spec_defaults() -> None:
    assert INPUT_SPEC.preferred_fps == 60
    assert INPUT_SPEC.minimum_fps == 30
    assert INPUT_SPEC.angle == "behind-server"
    assert INPUT_SPEC.mount == "tripod"
    assert INPUT_SPEC.file_extension == ".mp4"


def test_metrics_are_defined() -> None:
    assert len(PHASED_METRICS) >= 5
    assert "serve_avg_speed" in PHASED_METRICS
    assert "landing_point_error" in PHASED_METRICS
