"""Extract frames from side-angle clips and auto-label with the current model.

High-confidence detections become pseudo-labels for retraining.
Low-confidence or no-detection frames are included as hard negatives.

Usage:
    python3 -m scripts.extract_and_pseudolabel
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2

from src.detection.ball_detector import BallDetector


def extract_and_label(
    videos_dir: Path,
    output_root: Path,
    model_path: Path,
    conf_threshold: float = 0.25,
    every_n_frames: int = 3,
    train_ratio: float = 0.85,
) -> None:
    detector = BallDetector(
        model_path=str(model_path),
        class_id=0,
        conf_threshold=conf_threshold,
    )

    train_img_dir = output_root / "train" / "images"
    train_lbl_dir = output_root / "train" / "labels"
    val_img_dir = output_root / "valid" / "images"
    val_lbl_dir = output_root / "valid" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    video_files = sorted(videos_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos in {videos_dir}")

    total_frames_saved = 0
    total_labeled = 0

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Skip (cannot open): {video_path.name}")
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_name = video_path.stem

        frame_idx = 0
        clip_saved = 0
        clip_labeled = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % every_n_frames != 0:
                frame_idx += 1
                continue

            # Resize if too large (keep aspect ratio, max 1280 on long side)
            h_orig, w_orig = frame.shape[:2]
            max_dim = max(h_orig, w_orig)
            if max_dim > 1280:
                scale = 1280 / max_dim
                frame = cv2.resize(frame, (int(w_orig * scale), int(h_orig * scale)))
                width = frame.shape[1]
                height = frame.shape[0]

            # Pick train or val split
            is_train = random.random() < train_ratio
            img_dir = train_img_dir if is_train else val_img_dir
            lbl_dir = train_lbl_dir if is_train else val_lbl_dir

            img_name = f"{clip_name}_{frame_idx:06d}.jpg"
            img_path = img_dir / img_name
            lbl_path = lbl_dir / img_name.replace(".jpg", ".txt")

            # Run detection
            detections = detector.detect_frame(frame)

            # Save image
            cv2.imwrite(str(img_path), frame)

            # Write YOLO label (class x_center y_center width height, normalized)
            label_lines: list[str] = []
            for det in detections:
                cx = ((det.x1 + det.x2) / 2) / width
                cy = ((det.y1 + det.y2) / 2) / height
                bw = (det.x2 - det.x1) / width
                bh = (det.y2 - det.y1) / height
                label_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Write label file (empty file = hard negative, which is fine for YOLO)
            lbl_path.write_text("\n".join(label_lines), encoding="utf-8")

            clip_saved += 1
            if label_lines:
                clip_labeled += 1

            frame_idx += 1

        cap.release()
        total_frames_saved += clip_saved
        total_labeled += clip_labeled
        pct = 100 * clip_labeled / clip_saved if clip_saved else 0
        print(f"  {clip_name}: {clip_saved} frames, {clip_labeled} labeled ({pct:.0f}%)")

    print(f"\nTotal: {total_frames_saved} frames extracted, {total_labeled} with detections")
    print(f"Train: {len(list(train_img_dir.iterdir()))} images")
    print(f"Valid: {len(list(val_img_dir.iterdir()))} images")


def main() -> None:
    videos_dir = Path("data/raw/videos/side_angle")
    output_root = Path("data/processed/ball_dataset_v2")
    model_path = Path("models/volleyball_ball_yolo/weights/best.pt")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Train the base model first with: python3 -m scripts.setup_and_train")
        return

    print("=== Extracting frames + pseudo-labeling ===")
    print(f"Videos: {videos_dir}")
    print(f"Output: {output_root}")
    print(f"Model:  {model_path}")
    print()

    extract_and_label(videos_dir, output_root, model_path)

    # Write data.yaml
    yaml_path = Path("data/processed/ball_dataset_v2.yaml")
    yaml_path.write_text(
        f"nc: 1\n"
        f"names: ['ball']\n"
        f"train: {(output_root / 'train' / 'images').resolve()}\n"
        f"val: {(output_root / 'valid' / 'images').resolve()}\n",
        encoding="utf-8",
    )
    print(f"\nDataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
