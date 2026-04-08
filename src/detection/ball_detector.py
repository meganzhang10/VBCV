"""YOLO-based volleyball detector helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class BallDetection:
    """Single detection in XYXY pixel format."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class BallDetector:
    """Thin wrapper around Ultralytics YOLO for ball detection."""

    def __init__(self, model_path: str | Path, class_id: int = 0, conf_threshold: float = 0.2) -> None:
        self.model_path = str(model_path)
        self.class_id = class_id
        self.conf_threshold = conf_threshold
        self._model = self._load_model(self.model_path)

    @staticmethod
    def _load_model(model_path: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for detection. Install with: pip install ultralytics"
            ) from exc
        return YOLO(model_path)

    def detect_frame(self, frame_bgr: np.ndarray) -> list[BallDetection]:
        """Run inference on one BGR frame and return filtered detections."""
        results = self._model.predict(frame_bgr, conf=self.conf_threshold, verbose=False)
        if not results:
            return []
        boxes = results[0].boxes
        detections: list[BallDetection] = []
        if boxes is None:
            return detections
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id != self.class_id:
                continue
            conf = float(boxes.conf[i].item())
            xyxy = boxes.xyxy[i].tolist()
            detections.append(
                BallDetection(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                )
            )
        return detections

    def annotate_video(self, input_video: str | Path, output_video: str | Path) -> None:
        """Render a demo video with ball boxes."""
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = self.detect_frame(frame)
            for det in detections:
                p1 = (int(det.x1), int(det.y1))
                p2 = (int(det.x2), int(det.y2))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ball {det.confidence:.2f}",
                    (p1[0], max(20, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)

        writer.release()
        cap.release()
