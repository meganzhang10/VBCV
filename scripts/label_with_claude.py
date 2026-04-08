"""Auto-label volleyball frames using Claude's vision API.

Adapted from yolodex pipeline -- sends each frame to Claude, asks it to
locate the volleyball, converts response to YOLO format labels.

Usage:
    # Extract frames from videos first
    python3 -m scripts.label_with_claude extract \
        --videos-dir data/raw/videos/side_angle \
        --frames-dir data/processed/claude_labels/frames \
        --every-n 5

    # Label frames with Claude vision
    python3 -m scripts.label_with_claude label \
        --frames-dir data/processed/claude_labels/frames \
        --labels-dir data/processed/claude_labels/labels

    # Build YOLO dataset from labeled frames
    python3 -m scripts.label_with_claude build \
        --frames-dir data/processed/claude_labels/frames \
        --labels-dir data/processed/claude_labels/labels \
        --output-dir data/processed/ball_dataset_claude
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
import time
from pathlib import Path

import anthropic
import cv2


LABEL_PROMPT = """Look at this volleyball game frame. Find the volleyball (the ball being played, not any ball on the ground or held by someone).

If you can see the volleyball in play, return its bounding box as JSON:
{"found": true, "x": <left edge px>, "y": <top edge px>, "width": <width px>, "height": <height px>}

If there is no volleyball visible in play (between points, ball out of frame, replay graphic, etc.), return:
{"found": false}

Rules:
- Only detect the ball that is actively in play (being served, set, spiked, or flying through air)
- Do NOT detect balls sitting on the ground, held by players, or in the audience
- x,y is the TOP-LEFT corner of the bounding box in pixel coordinates
- The ball is typically small (10-40 pixels) at broadcast camera distance
- Return ONLY the JSON, no other text"""


def extract_frames(
    videos_dir: Path,
    frames_dir: Path,
    every_n: int = 5,
    max_frames_per_clip: int = 200,
) -> int:
    """Extract frames from all videos in a directory."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    video_files = sorted(videos_dir.glob("*.mp4"))
    total = 0

    for video_path in video_files:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  skip (cannot open): {video_path.name}")
            continue

        clip_name = video_path.stem
        frame_idx = 0
        clip_count = 0

        while clip_count < max_frames_per_clip:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % every_n == 0:
                # Resize to max 1280 on long side
                h, w = frame.shape[:2]
                if max(h, w) > 1280:
                    scale = 1280 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                out_path = frames_dir / f"{clip_name}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                clip_count += 1
                total += 1

            frame_idx += 1

        cap.release()
        print(f"  {clip_name}: {clip_count} frames")

    print(f"\nTotal: {total} frames extracted to {frames_dir}")
    return total


def encode_image(path: Path) -> str:
    """Encode image to base64 for Claude API."""
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def parse_json_response(text: str) -> dict:
    """Extract JSON from Claude's response."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding any JSON object
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"found": False}


def label_frame(client: anthropic.Anthropic, frame_path: Path) -> dict:
    """Send a frame to Claude and get volleyball bounding box."""
    img_b64 = encode_image(frame_path)
    ext = frame_path.suffix.lower()
    media_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": LABEL_PROMPT,
                    },
                ],
            }
        ],
    )

    return parse_json_response(response.content[0].text)


def label_frames(
    frames_dir: Path,
    labels_dir: Path,
    batch_size: int = 50,
    delay: float = 0.5,
) -> tuple[int, int]:
    """Label all frames in a directory using Claude vision."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    frames = sorted(frames_dir.glob("*.jpg"))
    total = len(frames)
    labeled = 0
    skipped = 0

    print(f"Labeling {total} frames with Claude Haiku vision...")
    print()

    for i, frame_path in enumerate(frames):
        label_path = labels_dir / frame_path.with_suffix(".txt").name

        # Skip if already labeled
        if label_path.exists():
            skipped += 1
            continue

        try:
            result = label_frame(client, frame_path)
        except Exception as e:
            print(f"  ERROR on {frame_path.name}: {e}")
            # Write empty label (hard negative)
            label_path.write_text("", encoding="utf-8")
            time.sleep(delay)
            continue

        if result.get("found"):
            # Get image dimensions for YOLO normalization
            img = cv2.imread(str(frame_path))
            if img is None:
                label_path.write_text("", encoding="utf-8")
                continue

            img_h, img_w = img.shape[:2]
            x = float(result["x"])
            y = float(result["y"])
            w = float(result["width"])
            h = float(result["height"])

            # Convert to YOLO format (center_x, center_y, width, height, normalized)
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Sanity check: ball should be small
            if nw < 0.15 and nh < 0.15 and nw > 0.003 and nh > 0.003:
                label_path.write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")
                labeled += 1
            else:
                # Too big or too small -- probably not actually the ball
                label_path.write_text("", encoding="utf-8")
        else:
            # No ball found -- empty label (hard negative for YOLO)
            label_path.write_text("", encoding="utf-8")

        if (i + 1) % 10 == 0 or i + 1 == total:
            pct = 100 * (i + 1) / total
            print(f"  [{i + 1}/{total}] ({pct:.0f}%) -- {labeled} balls found so far")

        time.sleep(delay)

    if skipped:
        print(f"\nSkipped {skipped} already-labeled frames")
    print(f"Done: {labeled}/{total} frames have ball labels")
    return labeled, total


def build_dataset(
    frames_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.85,
) -> None:
    """Organize labeled frames into YOLO train/val dataset."""
    train_img = output_dir / "train" / "images"
    train_lbl = output_dir / "train" / "labels"
    val_img = output_dir / "valid" / "images"
    val_lbl = output_dir / "valid" / "labels"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    frames = sorted(frames_dir.glob("*.jpg"))
    random.shuffle(frames)

    copied = 0
    for frame in frames:
        label = labels_dir / frame.with_suffix(".txt").name
        if not label.exists():
            continue

        is_train = random.random() < train_ratio
        img_dst = train_img if is_train else val_img
        lbl_dst = train_lbl if is_train else val_lbl

        import shutil
        shutil.copy2(frame, img_dst / frame.name)
        shutil.copy2(label, lbl_dst / label.name)
        copied += 1

    # Write data.yaml
    yaml_path = output_dir.parent / f"{output_dir.name}.yaml"
    yaml_path.write_text(
        f"nc: 1\n"
        f"names: ['ball']\n"
        f"train: {train_img.resolve()}\n"
        f"val: {val_img.resolve()}\n",
        encoding="utf-8",
    )

    train_count = len(list(train_img.iterdir()))
    val_count = len(list(val_img.iterdir()))
    print(f"Dataset built: {train_count} train, {val_count} val")
    print(f"YAML: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label volleyball with Claude vision")
    sub = parser.add_subparsers(dest="command")

    ext = sub.add_parser("extract", help="Extract frames from videos")
    ext.add_argument("--videos-dir", required=True)
    ext.add_argument("--frames-dir", required=True)
    ext.add_argument("--every-n", type=int, default=5)
    ext.add_argument("--max-frames", type=int, default=200)

    lbl = sub.add_parser("label", help="Label frames with Claude vision")
    lbl.add_argument("--frames-dir", required=True)
    lbl.add_argument("--labels-dir", required=True)
    lbl.add_argument("--delay", type=float, default=0.3)

    bld = sub.add_parser("build", help="Build YOLO dataset from labeled frames")
    bld.add_argument("--frames-dir", required=True)
    bld.add_argument("--labels-dir", required=True)
    bld.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    if args.command == "extract":
        extract_frames(Path(args.videos_dir), Path(args.frames_dir), args.every_n, args.max_frames)
    elif args.command == "label":
        label_frames(Path(args.frames_dir), Path(args.labels_dir), delay=args.delay)
    elif args.command == "build":
        build_dataset(Path(args.frames_dir), Path(args.labels_dir), Path(args.output_dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
