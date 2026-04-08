"""Kalman-filter-based ball trajectory predictor with visualization.

Provides real-time ball tracking with:
- Bounding box on current ball position
- Solid fading line for past tracked path
- Dotted line for predicted future trajectory
- Glow effect on ball centroid
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TrajectoryConfig:
    """Tuning knobs for the predictor + renderer."""

    # Kalman filter noise
    process_noise: float = 1e-2
    measurement_noise: float = 1e-1

    # Visualization
    past_trail_length: int = 30
    future_frames: int = 15
    past_color_start: tuple[int, int, int] = (100, 100, 100)  # gray (oldest)
    past_color_end: tuple[int, int, int] = (0, 255, 255)     # bright yellow (newest)
    future_color: tuple[int, int, int] = (0, 100, 255)       # orange
    bbox_color: tuple[int, int, int] = (0, 255, 0)           # green
    glow_color: tuple[int, int, int] = (0, 255, 255)         # yellow


class BallTrajectoryPredictor:
    """6D Kalman filter (x, y, vx, vy, ax, ay) for ball trajectory prediction."""

    def __init__(self, config: TrajectoryConfig | None = None) -> None:
        self.config = config or TrajectoryConfig()
        self.kf = cv2.KalmanFilter(6, 2)
        self._setup_kalman()
        self.past_positions: list[tuple[int, int]] = []
        self.initialized: bool = False
        self.frames_since_detection: int = 0

    def _setup_kalman(self) -> None:
        dt = 1.0  # one frame timestep

        # State transition: constant acceleration model
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5 * dt**2, 0],
            [0, 1, 0, dt, 0, 0.5 * dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=np.float32)

        # We only observe (x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)

        # Tuned noise: low process noise = smoother predictions, less flicker
        # Higher measurement noise = trust the model's momentum more than individual detections
        pn = np.eye(6, dtype=np.float32) * self.config.process_noise
        # Position noise lower, velocity/accel noise even lower for stability
        pn[0, 0] = self.config.process_noise
        pn[1, 1] = self.config.process_noise
        pn[2, 2] = self.config.process_noise * 0.5  # velocity x
        pn[3, 3] = self.config.process_noise * 0.5  # velocity y
        pn[4, 4] = self.config.process_noise * 0.1  # acceleration x
        pn[5, 5] = self.config.process_noise * 0.1  # acceleration y
        self.kf.processNoiseCov = pn
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.config.measurement_noise

    def reset(self) -> None:
        """Reset the filter state."""
        self.kf = cv2.KalmanFilter(6, 2)
        self._setup_kalman()
        self.past_positions.clear()
        self.initialized = False
        self.frames_since_detection = 0

    def update(self, centroid: tuple[int, int] | None) -> None:
        """Feed a detection centroid (x, y) or None if no detection this frame."""
        if centroid is not None:
            measurement = np.array(
                [[np.float32(centroid[0])], [np.float32(centroid[1])]],
                dtype=np.float32,
            )

            if not self.initialized:
                self.kf.statePre = np.array(
                    [[centroid[0]], [centroid[1]], [0], [0], [0], [0]],
                    dtype=np.float32,
                )
                self.kf.statePost = self.kf.statePre.copy()
                self.initialized = True

            self.kf.correct(measurement)
            self.past_positions.append(centroid)
            self.past_positions = self.past_positions[-self.config.past_trail_length:]
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1

        if self.initialized:
            self.kf.predict()

    def predict_future(self, n_frames: int | None = None, min_detections: int = 5) -> list[tuple[int, int]]:
        """Predict next N positions without modifying filter state.

        Only returns predictions if we have enough past detections for a
        stable velocity estimate. This prevents flickering predictions
        from 1-2 noisy detections.
        """
        if not self.initialized:
            return []

        # Don't predict if we don't have enough data for stable velocity
        if len(self.past_positions) < min_detections:
            return []

        # Don't predict if we haven't seen the ball in a while
        # 30 frames = ~1 sec at 30fps -- keep predicting through short gaps
        if self.frames_since_detection > 30:
            return []

        n = n_frames or self.config.future_frames
        points: list[tuple[int, int]] = []
        state = self.kf.statePost.copy()
        transition = self.kf.transitionMatrix

        for _ in range(n):
            state = transition @ state
            x = int(state[0, 0])
            y = int(state[1, 0])
            points.append((x, y))

        return points

    @property
    def current_velocity(self) -> tuple[float, float] | None:
        """Return (vx, vy) in pixels/frame from the filter state."""
        if not self.initialized:
            return None
        return (float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0]))

    @property
    def current_speed_px(self) -> float | None:
        """Return speed in pixels/frame."""
        vel = self.current_velocity
        if vel is None:
            return None
        return float(np.hypot(vel[0], vel[1]))


def draw_trajectory_overlay(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    past: list[tuple[int, int]],
    future: list[tuple[int, int]],
    config: TrajectoryConfig | None = None,
) -> np.ndarray:
    """Render trajectory visualization onto a frame.

    Args:
        frame: BGR image to draw on (will be modified in place).
        bbox: Current ball bounding box (x1, y1, x2, y2) or None.
        past: List of past centroid positions.
        future: List of predicted future positions.
        config: Visualization config.

    Returns:
        The frame with trajectory overlay drawn.
    """
    cfg = config or TrajectoryConfig()
    h, w = frame.shape[:2]

    def _in_bounds(pt: tuple[int, int]) -> bool:
        return 0 <= pt[0] < w and 0 <= pt[1] < h

    # --- Past path: solid line with color/thickness fade ---
    if len(past) >= 2:
        for i in range(1, len(past)):
            if not _in_bounds(past[i - 1]) or not _in_bounds(past[i]):
                continue
            alpha = i / len(past)
            color = tuple(
                int(cfg.past_color_start[c] + alpha * (cfg.past_color_end[c] - cfg.past_color_start[c]))
                for c in range(3)
            )
            thickness = max(1, int(3 * alpha))
            cv2.line(frame, past[i - 1], past[i], color, thickness, cv2.LINE_AA)

        # Small circles at each past position for breadcrumb effect
        for i, pt in enumerate(past):
            if not _in_bounds(pt):
                continue
            alpha = (i + 1) / len(past)
            radius = max(1, int(3 * alpha))
            color = tuple(
                int(cfg.past_color_start[c] + alpha * (cfg.past_color_end[c] - cfg.past_color_start[c]))
                for c in range(3)
            )
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)

    # --- Future path: dotted/dashed line ---
    if len(future) >= 2:
        for i in range(1, len(future)):
            if not _in_bounds(future[i]):
                break
            alpha = 1.0 - (i / len(future))
            if alpha < 0.1:
                break

            # Draw dashed segments (every other)
            if i % 2 == 1:
                intensity = int(255 * alpha)
                color = (
                    int(cfg.future_color[0] * alpha),
                    int(cfg.future_color[1] * alpha),
                    int(cfg.future_color[2] * alpha),
                )
                cv2.line(frame, future[i - 1], future[i], color, 2, cv2.LINE_AA)

            # Dots at each predicted point
            dot_intensity = max(30, int(200 * alpha))
            cv2.circle(
                frame,
                future[i],
                max(2, int(4 * alpha)),
                (0, dot_intensity, int(cfg.future_color[2] * alpha)),
                -1,
                cv2.LINE_AA,
            )

    # --- Current bounding box + glow ---
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), cfg.bbox_color, 2, cv2.LINE_AA)

        # Bright marker at centroid (no full-frame glow that dims everything)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # Outer glow ring
        cv2.circle(frame, (cx, cy), 12, cfg.glow_color, 2, cv2.LINE_AA)
        # Inner bright dot
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)

    return frame


def draw_speed_label(
    frame: np.ndarray,
    speed_px_per_frame: float | None,
    fps: float,
    position: tuple[int, int] = (20, 40),
    px_per_meter: float | None = None,
) -> np.ndarray:
    """Draw estimated speed label on frame.

    If px_per_meter is provided, shows speed in km/h.
    Otherwise estimates it from frame width assuming a standard
    volleyball court (9m wide) fills ~70% of a broadcast frame.
    """
    if speed_px_per_frame is None:
        return frame

    # px/frame -> px/sec
    speed_px_sec = speed_px_per_frame * fps

    # Estimate px_per_meter from frame width if not provided
    # Standard volleyball court = 9m wide, typically fills ~70% of broadcast frame
    if px_per_meter is None:
        frame_width = frame.shape[1]
        court_width_px = frame_width * 0.70
        px_per_meter = court_width_px / 9.0

    # Convert to m/s then km/h
    speed_mps = speed_px_sec / px_per_meter
    speed_kph = speed_mps * 3.6

    label = f"Ball speed: {speed_kph:.0f} km/h"

    cv2.putText(
        frame,
        label,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame
