"""Project-level defaults for Sprint 0 assumptions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class InputSpec:
    preferred_fps: int = 60
    minimum_fps: int = 30
    preferred_resolution: tuple[int, int] = (1920, 1080)
    angle: str = "behind-server"
    mount: str = "tripod"
    file_extension: str = ".mp4"
    codec_family: str = "h264"


INPUT_SPEC = InputSpec()

PHASED_METRICS = [
    "ball_detection_precision",
    "ball_detection_recall",
    "tracking_continuity",
    "speed_estimate_stability",
    "landing_point_error",
    "serve_avg_speed",
    "serve_max_speed",
    "serve_in_out_ratio",
    "landing_heatmap_density",
]
