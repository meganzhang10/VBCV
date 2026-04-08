"""Court-plane calibration and coordinate conversion for Sprint 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PointPair:
    name: str
    pixel: tuple[float, float]
    court: tuple[float, float]


@dataclass
class CalibrationResult:
    homography: np.ndarray
    inverse_homography: np.ndarray
    points: list[PointPair]

    def pixel_to_court(self, x_pixel: float, y_pixel: float) -> tuple[float, float]:
        src = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(src, self.homography)[0][0]
        return float(mapped[0]), float(mapped[1])

    def court_to_pixel(self, x_court: float, y_court: float) -> tuple[float, float]:
        src = np.array([[[x_court, y_court]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(src, self.inverse_homography)[0][0]
        return float(mapped[0]), float(mapped[1])


def load_point_pairs(points_json: str | Path) -> list[PointPair]:
    payload = json.loads(Path(points_json).read_text(encoding="utf-8"))
    raw_points = payload.get("points", [])
    if len(raw_points) < 4:
        raise ValueError("Need at least 4 point correspondences for homography.")
    points: list[PointPair] = []
    for item in raw_points:
        pixel = item["pixel"]
        court = item["court"]
        points.append(
            PointPair(
                name=str(item.get("name", f"p{len(points)}")),
                pixel=(float(pixel[0]), float(pixel[1])),
                court=(float(court[0]), float(court[1])),
            )
        )
    return points


def estimate_homography(points: list[PointPair]) -> CalibrationResult:
    if len(points) < 4:
        raise ValueError("Need at least 4 point correspondences for homography.")
    pixel = np.array([[p.pixel[0], p.pixel[1]] for p in points], dtype=np.float32)
    court = np.array([[p.court[0], p.court[1]] for p in points], dtype=np.float32)
    homography, _ = cv2.findHomography(pixel, court, method=0)
    if homography is None:
        raise RuntimeError("Failed to estimate homography.")
    inverse_h = np.linalg.inv(homography)
    return CalibrationResult(homography=homography, inverse_homography=inverse_h, points=points)


def write_points_template(out_path: str | Path) -> None:
    """Create a starter point file using standard indoor court coordinates in meters."""
    template = {
        "court_units": "meters",
        "court_reference": "indoor_volleyball_18x9",
        "notes": (
            "Fill pixel coordinates from your frame. "
            "Keep court coords fixed unless you use a different coordinate system."
        ),
        "points": [
            {"name": "near_left_endline_corner", "pixel": [0, 0], "court": [0.0, 0.0]},
            {"name": "near_right_endline_corner", "pixel": [0, 0], "court": [0.0, 9.0]},
            {"name": "far_left_endline_corner", "pixel": [0, 0], "court": [18.0, 0.0]},
            {"name": "far_right_endline_corner", "pixel": [0, 0], "court": [18.0, 9.0]},
            {"name": "left_attack_line_near", "pixel": [0, 0], "court": [6.0, 0.0]},
            {"name": "right_attack_line_near", "pixel": [0, 0], "court": [6.0, 9.0]},
            {"name": "left_attack_line_far", "pixel": [0, 0], "court": [12.0, 0.0]},
            {"name": "right_attack_line_far", "pixel": [0, 0], "court": [12.0, 9.0]},
        ],
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(template, indent=2), encoding="utf-8")


def _project_track_to_court(
    track_csv: str | Path,
    calibration: CalibrationResult,
) -> pd.DataFrame:
    df = pd.read_csv(track_csv)
    required = {"frame_index", "timestamp_sec"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Track CSV missing required columns: {missing}")
    if "smooth_x" not in df.columns or "smooth_y" not in df.columns:
        raise ValueError("Track CSV missing smooth_x/smooth_y columns from Sprint 2.")

    px = df["smooth_x"].copy()
    py = df["smooth_y"].copy()
    if "raw_x" in df.columns:
        px = px.where(~px.isna(), df["raw_x"])
    if "raw_y" in df.columns:
        py = py.where(~py.isna(), df["raw_y"])

    court_x: list[float | None] = []
    court_y: list[float | None] = []
    for x, y in zip(px, py):
        if pd.isna(x) or pd.isna(y):
            court_x.append(None)
            court_y.append(None)
            continue
        cx, cy = calibration.pixel_to_court(float(x), float(y))
        court_x.append(cx)
        court_y.append(cy)

    output = df.copy()
    output["court_x_m"] = court_x
    output["court_y_m"] = court_y
    return output


def estimate_landing_point(track_with_court: pd.DataFrame) -> tuple[float, float] | None:
    """Approximate landing as the last valid tracked court-plane point."""
    valid = track_with_court.dropna(subset=["court_x_m", "court_y_m"])
    if valid.empty:
        return None
    last = valid.iloc[-1]
    return float(last["court_x_m"]), float(last["court_y_m"])


def estimate_speed_mps(track_with_court: pd.DataFrame) -> float | None:
    """Approximate 2D court-plane speed from frame-to-frame displacement."""
    valid = track_with_court.dropna(subset=["court_x_m", "court_y_m"])
    if len(valid) < 2:
        return None
    dx = valid["court_x_m"].diff()
    dy = valid["court_y_m"].diff()
    dt = valid["timestamp_sec"].diff()
    step_dist = np.sqrt(dx * dx + dy * dy)
    speed = step_dist / dt.replace(0, np.nan)
    speed = speed.replace([np.inf, -np.inf], np.nan).dropna()
    if speed.empty:
        return None
    return float(speed.median())


def export_projected_track(track_df: pd.DataFrame, out_csv: str | Path) -> None:
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    track_df.to_csv(path, index=False)


def export_calibration_report(
    calibration: CalibrationResult,
    landing_point_m: tuple[float, float] | None,
    speed_mps: float | None,
    out_json: str | Path,
) -> None:
    path = Path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "assumptions": [
            "Ball motion is projected to the court plane via homography.",
            "Landing point is approximated as last valid tracked point in clip.",
            "Speed is 2D court-plane speed, not true 3D ball speed.",
        ],
        "homography": calibration.homography.tolist(),
        "inverse_homography": calibration.inverse_homography.tolist(),
        "points": [asdict(p) for p in calibration.points],
        "estimated_landing_point_m": landing_point_m,
        "estimated_speed_mps": speed_mps,
        "estimated_speed_kph": None if speed_mps is None else speed_mps * 3.6,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def draw_topdown_court_overlay(
    landing_point_m: tuple[float, float] | None,
    out_image: str | Path,
    court_length_m: float = 18.0,
    court_width_m: float = 9.0,
    pixels_per_meter: int = 60,
) -> None:
    margin = 40
    width = int(court_length_m * pixels_per_meter) + 2 * margin
    height = int(court_width_m * pixels_per_meter) + 2 * margin
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    def to_px(x_m: float, y_m: float) -> tuple[int, int]:
        x = int(margin + x_m * pixels_per_meter)
        y = int(margin + y_m * pixels_per_meter)
        return x, y

    near_left = to_px(0.0, 0.0)
    far_left = to_px(court_length_m, 0.0)
    far_right = to_px(court_length_m, court_width_m)

    cv2.rectangle(image, near_left, far_right, (30, 30, 30), 2)
    cv2.line(image, to_px(court_length_m / 2.0, 0.0), to_px(court_length_m / 2.0, court_width_m), (0, 0, 0), 2)
    cv2.line(image, to_px(6.0, 0.0), to_px(6.0, court_width_m), (80, 80, 80), 1)
    cv2.line(image, to_px(12.0, 0.0), to_px(12.0, court_width_m), (80, 80, 80), 1)

    cv2.putText(image, "Near Endline", (near_left[0], near_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "Far Endline", (far_left[0], far_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if landing_point_m is not None:
        lx, ly = landing_point_m
        if 0 <= lx <= court_length_m and 0 <= ly <= court_width_m:
            cv2.circle(image, to_px(lx, ly), 8, (0, 0, 255), -1)
            cv2.putText(
                image,
                f"Landing ~ ({lx:.2f}m, {ly:.2f}m)",
                (margin, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        else:
            cv2.putText(
                image,
                f"Landing estimated outside court: ({lx:.2f}m, {ly:.2f}m)",
                (margin, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

    path = Path(out_image)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def calibrate_and_project(
    points_json: str | Path,
    track_csv: str | Path,
) -> tuple[CalibrationResult, pd.DataFrame, tuple[float, float] | None, float | None]:
    points = load_point_pairs(points_json)
    calibration = estimate_homography(points)
    projected = _project_track_to_court(track_csv, calibration)
    landing = estimate_landing_point(projected)
    speed = estimate_speed_mps(projected)
    return calibration, projected, landing, speed
