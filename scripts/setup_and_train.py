"""Download volleyball dataset from Roboflow and fine-tune YOLOv8 for ball detection.

Usage:
    # Set your free Roboflow API key (https://app.roboflow.com/settings/api)
    export ROBOFLOW_API_KEY="your_key_here"

    # Run setup + training
    python3 -m scripts.setup_and_train

    # Or just download the dataset without training
    python3 -m scripts.setup_and_train --download-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def download_dataset(api_key: str, dest: Path) -> Path:
    """Download volleyball ball detection dataset from Roboflow in YOLOv8 format."""
    from roboflow import Roboflow

    print("[1/3] Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)

    print("[2/3] Fetching volleyball ball detection dataset (548 images, YOLOv8 format)...")
    project = rf.workspace("primaryws").project("volleyball_ball_object_detection_dataset")
    version = project.version(1)

    print(f"[3/3] Downloading to {dest}...")
    dataset = version.download("yolov8", location=str(dest))
    print(f"Dataset downloaded to: {dataset.location}")
    return Path(dataset.location)


def prepare_dataset_yaml(dataset_dir: Path, output_yaml: Path) -> None:
    """Build a clean data.yaml with absolute paths by scanning the dataset directory."""
    dataset_abs = dataset_dir.resolve()

    # Detect available splits (roboflow uses "valid" not "val")
    split_map: dict[str, str] = {}
    for yaml_key, dir_names in [("train", ["train"]), ("val", ["valid", "val"]), ("test", ["test"])]:
        for name in dir_names:
            images_dir = dataset_abs / name / "images"
            if images_dir.is_dir() and any(images_dir.iterdir()):
                split_map[yaml_key] = str(images_dir)
                break

    if "train" not in split_map:
        raise FileNotFoundError(f"No train/images directory found in {dataset_abs}")

    # Read class info from roboflow's data.yaml if it exists
    source_yaml = dataset_abs / "data.yaml"
    names = ["ball"]
    if source_yaml.exists():
        import yaml  # type: ignore[import-untyped]
        with open(source_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "names" in data:
            names = data["names"]

    lines = [f"nc: {len(names)}", f"names: {names}"]
    for key in ("train", "val", "test"):
        if key in split_map:
            lines.append(f"{key}: {split_map[key]}")

    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Prepared data.yaml at {output_yaml}")
    for key, path in split_map.items():
        count = len(list(Path(path).iterdir()))
        print(f"  {key}: {count} images")


def train_model(data_yaml: Path, output_dir: Path) -> Path:
    """Fine-tune YOLOv8n on the volleyball dataset."""
    from ultralytics import YOLO

    print("\n=== Starting YOLOv8n fine-tuning ===")
    print(f"Dataset: {data_yaml}")
    print(f"Output:  {output_dir}")
    print()

    # Start from pretrained YOLOv8n (auto-downloads ~6MB from ultralytics hub)
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=50,
        batch=8,
        project=str(output_dir),
        name="volleyball_ball_yolo",
        exist_ok=True,
        patience=10,
        verbose=True,
    )

    weights_path = output_dir / "volleyball_ball_yolo" / "weights" / "best.pt"
    if weights_path.exists():
        # Also copy to the expected location for the rest of the pipeline
        final_path = Path("models/volleyball_ball_yolo/weights/best.pt")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights_path, final_path)
        print(f"\nTraining complete! Best weights copied to: {final_path}")
        return final_path
    else:
        print(f"\nTraining finished but best.pt not found at {weights_path}")
        print("Check the training output directory for results.")
        return weights_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset and train volleyball ball detector")
    parser.add_argument("--download-only", action="store_true", help="Only download dataset, skip training")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("ERROR: Roboflow API key required.")
        print("  Get a free key at: https://app.roboflow.com/settings/api")
        print("  Then either:")
        print("    export ROBOFLOW_API_KEY='your_key'")
        print("    python3 -m scripts.setup_and_train --api-key your_key")
        sys.exit(1)

    dataset_dest = Path("data/processed/ball_dataset")
    data_yaml = Path("data/processed/ball_dataset.yaml")

    if dataset_dest.exists() and any(dataset_dest.iterdir()):
        print(f"Dataset already exists at {dataset_dest}, skipping download.")
        print("Delete the directory to re-download.")
    else:
        dataset_dir = download_dataset(api_key, dataset_dest)

    prepare_dataset_yaml(dataset_dest, data_yaml)

    if args.download_only:
        print("\n--download-only set. Skipping training.")
        print(f"Dataset ready at: {dataset_dest}")
        print(f"Data YAML at:     {data_yaml}")
        print("\nTo train manually:")
        print(f"  python3 -m scripts.setup_and_train")
        return

    print("\n" + "=" * 60)
    print("Dataset ready. Starting training...")
    print("=" * 60)

    train_model(data_yaml, Path("models"))


if __name__ == "__main__":
    main()
