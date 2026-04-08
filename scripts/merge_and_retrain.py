"""Merge original Roboflow dataset with pseudo-labeled frames and retrain.

Usage:
    python3 -m scripts.merge_and_retrain
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2


def merge_datasets(
    original: Path,
    pseudo: Path,
    merged: Path,
) -> None:
    """Copy both datasets into one merged directory."""
    for split in ["train", "valid"]:
        for subdir in ["images", "labels"]:
            out = merged / split / subdir
            out.mkdir(parents=True, exist_ok=True)

            # Copy original
            src_orig = original / split / subdir
            if src_orig.exists():
                for f in src_orig.iterdir():
                    dst = out / f"orig_{f.name}"
                    if not dst.exists():
                        shutil.copy2(f, dst)

            # Copy pseudo-labeled
            src_pseudo = pseudo / split / subdir
            if src_pseudo.exists():
                for f in src_pseudo.iterdir():
                    dst = out / f"pseudo_{f.name}"
                    if not dst.exists():
                        shutil.copy2(f, dst)

    # Count
    for split in ["train", "valid"]:
        img_count = len(list((merged / split / "images").iterdir()))
        lbl_count = len(list((merged / split / "labels").iterdir()))
        print(f"  {split}: {img_count} images, {lbl_count} labels")


def train(data_yaml: Path, output_dir: Path, epochs: int = 60, batch: int = 8) -> None:
    from ultralytics import YOLO

    # Start from our previously trained model (transfer learning)
    base_model = Path("models/volleyball_ball_yolo/weights/best.pt")
    if base_model.exists():
        print(f"Fine-tuning from: {base_model}")
        model = YOLO(str(base_model))
    else:
        print("Starting from pretrained yolov8n.pt")
        model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=epochs,
        batch=batch,
        project=str(output_dir),
        name="volleyball_ball_yolo_v2",
        exist_ok=True,
        patience=15,
        verbose=True,
        lr0=0.005,  # Lower LR since we're fine-tuning
    )

    best = output_dir / "volleyball_ball_yolo_v2" / "weights" / "best.pt"
    if best.exists():
        final = Path("models/volleyball_ball_yolo/weights/best.pt")
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, final)
        print(f"\nBest weights saved to: {final}")


def main() -> None:
    original = Path("data/processed/ball_dataset")
    pseudo = Path("data/processed/ball_dataset_v2")
    merged = Path("data/processed/ball_dataset_merged")

    print("=== Merging datasets ===")
    merge_datasets(original, pseudo, merged)

    yaml_path = Path("data/processed/ball_dataset_merged.yaml")
    yaml_path.write_text(
        f"nc: 1\n"
        f"names: ['ball']\n"
        f"train: {(merged / 'train' / 'images').resolve()}\n"
        f"val: {(merged / 'valid' / 'images').resolve()}\n",
        encoding="utf-8",
    )
    print(f"\nDataset YAML: {yaml_path}")

    print("\n=== Starting retraining ===")
    train(yaml_path, Path("models"))


if __name__ == "__main__":
    main()
