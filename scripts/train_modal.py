"""Train YOLO volleyball detector on Modal (cloud GPU).

Usage:
    modal run scripts/train_modal.py

Runs on a T4 GPU in the cloud. Training takes ~2-3 min for 987 images.
Downloads best.pt to local machine when done.
"""

from __future__ import annotations

from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.2.0",
        "opencv-python-headless>=4.9.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
    )
)

app = modal.App("vbcv-yolo-training", image=image)

volume = modal.Volume.from_name("vbcv-training-data", create_if_missing=True)


@app.function(gpu="T4", timeout=600, volumes={"/data": volume})
def train_yolo(dataset_bytes: bytes) -> bytes:
    """Train YOLOv8n on volleyball data and return the best weights."""
    import zipfile
    import io
    from pathlib import Path
    from ultralytics import YOLO

    # Extract dataset
    with zipfile.ZipFile(io.BytesIO(dataset_bytes)) as zf:
        zf.extractall("/work")

    # Write dataset yaml
    train_dir = Path("/work/data/processed/ball_dataset_combined/train/images")
    val_dir = Path("/work/data/processed/ball_dataset_combined/valid/images")

    print(f"Train images: {len(list(train_dir.iterdir()))}")
    print(f"Val images: {len(list(val_dir.iterdir()))}")

    yaml_path = Path("/work/dataset.yaml")
    yaml_path.write_text(
        f"nc: 1\n"
        f"names: ['volleyball']\n"
        f"train: {train_dir}\n"
        f"val: {val_dir}\n"
    )

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        imgsz=640,
        epochs=80,
        batch=32,
        patience=15,
        cache=True,
        device=0,
        verbose=True,
        project="/work/results",
        name="volleyball",
        exist_ok=True,
    )

    best_pt = Path("/work/results/volleyball/weights/best.pt")
    if not best_pt.exists():
        raise FileNotFoundError("Training failed -- no best.pt found")

    size_mb = best_pt.stat().st_size / 1024 / 1024
    print(f"Training complete! best.pt size: {size_mb:.1f} MB")
    return best_pt.read_bytes()


@app.local_entrypoint()
def main() -> None:
    import zipfile
    import io

    print("Zipping training data...")
    buf = io.BytesIO()
    src = Path("/tmp/VBCV/data/processed/ball_dataset_combined")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in src.rglob("*"):
            if f.is_file():
                zf.write(f, f"data/processed/ball_dataset_combined/{f.relative_to(src)}")

    dataset_bytes = buf.getvalue()
    print(f"Dataset: {len(dataset_bytes) / 1024 / 1024:.1f} MB")

    print("Uploading + training on Modal T4 GPU...")
    weights_bytes = train_yolo.remote(dataset_bytes)

    for dst in [
        Path("/tmp/VBCV/models/volleyball_ball_yolo/weights/best.pt"),
        Path("/Users/stephenhung/Documents/GitHub/VBCV/models/volleyball_ball_yolo/weights/best.pt"),
    ]:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(weights_bytes)

    print(f"Saved best.pt ({len(weights_bytes) / 1024 / 1024:.1f} MB)")
    print("Done!")
