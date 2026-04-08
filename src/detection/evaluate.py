"""Evaluation helpers for object detection on YOLO labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.detection.ball_detector import BallDetection, BallDetector


@dataclass(frozen=True)
class EvalMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def _iou(a: BallDetection, b: BallDetection) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_yolo_label(label_path: Path, image_width: int, image_height: int) -> list[BallDetection]:
    if not label_path.exists():
        return []
    detections: list[BallDetection] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        cx = float(parts[1]) * image_width
        cy = float(parts[2]) * image_height
        w = float(parts[3]) * image_width
        h = float(parts[4]) * image_height
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        detections.append(BallDetection(x1, y1, x2, y2, 1.0, cls_id))
    return detections


def evaluate_image_folder(
    detector: BallDetector,
    images_dir: str | Path,
    labels_dir: str | Path,
    iou_threshold: float = 0.3,
) -> EvalMetrics:
    import cv2

    image_paths = sorted(Path(images_dir).glob("*.jpg"))
    labels_base = Path(labels_dir)

    tp = 0
    fp = 0
    fn = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt = load_yolo_label(labels_base / f"{image_path.stem}.txt", w, h)
        preds = detector.detect_frame(image)

        matched_gt: set[int] = set()
        for pred in sorted(preds, key=lambda x: x.confidence, reverse=True):
            best_idx = -1
            best_iou = 0.0
            for i, g in enumerate(gt):
                if i in matched_gt or g.class_id != pred.class_id:
                    continue
                score = _iou(pred, g)
                if score > best_iou:
                    best_iou = score
                    best_idx = i
            if best_idx >= 0 and best_iou >= iou_threshold:
                matched_gt.add(best_idx)
                tp += 1
            else:
                fp += 1
        fn += max(0, len(gt) - len(matched_gt))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return EvalMetrics(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)
