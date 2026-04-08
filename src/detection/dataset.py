"""Dataset prep utilities for Sprint 1 ball detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd


@dataclass(frozen=True)
class ClipRecord:
    file_name: str
    source: str
    angle: str
    fps: int
    resolution: str
    status: str


def load_manifest(manifest_csv: str | Path) -> list[ClipRecord]:
    df = pd.read_csv(manifest_csv)
    required_columns = {"file_name", "source", "angle", "fps", "resolution", "status"}
    missing = required_columns - set(df.columns)
    if missing:
        missing_joined = ", ".join(sorted(missing))
        raise ValueError(f"Manifest missing required columns: {missing_joined}")

    records: list[ClipRecord] = []
    for row in df.itertuples(index=False):
        records.append(
            ClipRecord(
                file_name=str(row.file_name),
                source=str(row.source),
                angle=str(row.angle),
                fps=int(row.fps),
                resolution=str(row.resolution),
                status=str(row.status),
            )
        )
    return records


def _split_tag(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_cutoff = int(total * train_ratio)
    val_cutoff = train_cutoff + int(total * val_ratio)
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "val"
    return "test"


def extract_frames_from_manifest(
    manifest_csv: str | Path,
    videos_dir: str | Path,
    dataset_root: str | Path,
    every_n_frames: int = 3,
    max_frames_per_clip: int = 300,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> int:
    """Extract JPG frames split by clip into train/val/test folders."""
    records = load_manifest(manifest_csv)
    videos_dir_path = Path(videos_dir)
    dataset_root_path = Path(dataset_root)

    saved = 0
    for idx, record in enumerate(records):
        split = _split_tag(idx, len(records), train_ratio=train_ratio, val_ratio=val_ratio)
        target_dir = dataset_root_path / "images" / split
        target_dir.mkdir(parents=True, exist_ok=True)

        video_path = videos_dir_path / record.file_name
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        frame_idx = 0
        saved_for_clip = 0
        clip_name = Path(record.file_name).stem
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % every_n_frames == 0:
                out_name = f"{clip_name}_{frame_idx:06d}.jpg"
                out_path = target_dir / out_name
                cv2.imwrite(str(out_path), frame)
                saved += 1
                saved_for_clip += 1
                if saved_for_clip >= max_frames_per_clip:
                    break
            frame_idx += 1
        cap.release()

    for split in ("train", "val", "test"):
        (dataset_root_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    return saved


def write_yolo_dataset_yaml(dataset_root: str | Path, out_yaml: str | Path) -> Path:
    """Write a YOLO data YAML file for 1-class volleyball detection."""
    dataset_root_path = Path(dataset_root).resolve()
    out_yaml_path = Path(out_yaml)
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            f"path: {dataset_root_path}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: volleyball",
            "",
        ]
    )
    out_yaml_path.write_text(content, encoding="utf-8")
    return out_yaml_path
