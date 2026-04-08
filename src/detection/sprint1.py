"""Sprint 1 CLI: data prep, train, eval, and demo for volleyball ball detection."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.detection.ball_detector import BallDetector
from src.detection.dataset import extract_frames_from_manifest, write_yolo_dataset_yaml
from src.detection.evaluate import evaluate_image_folder


def cmd_extract(args: argparse.Namespace) -> None:
    extracted = extract_frames_from_manifest(
        manifest_csv=args.manifest,
        videos_dir=args.videos_dir,
        dataset_root=args.dataset_root,
        every_n_frames=args.every_n_frames,
        max_frames_per_clip=args.max_frames_per_clip,
    )
    print(f"Extracted {extracted} frames into {args.dataset_root}/images/*")
    yaml_path = write_yolo_dataset_yaml(args.dataset_root, args.data_yaml)
    print(f"Wrote YOLO dataset file: {yaml_path}")
    print("Next: label images and write YOLO txt files into labels/train|val|test")


def cmd_train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required for training.") from exc
    model = YOLO(args.base_model)
    model.train(
        data=args.data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project_dir,
        name=args.run_name,
    )
    print("Training finished.")


def cmd_eval(args: argparse.Namespace) -> None:
    detector = BallDetector(
        model_path=args.model_path,
        class_id=args.class_id,
        conf_threshold=args.conf_threshold,
    )
    metrics = evaluate_image_folder(
        detector=detector,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        iou_threshold=args.iou_threshold,
    )
    print(
        f"precision={metrics.precision:.3f} recall={metrics.recall:.3f} "
        f"f1={metrics.f1:.3f} tp={metrics.tp} fp={metrics.fp} fn={metrics.fn}"
    )


def cmd_demo(args: argparse.Namespace) -> None:
    Path(args.output_video).parent.mkdir(parents=True, exist_ok=True)
    detector = BallDetector(
        model_path=args.model_path,
        class_id=args.class_id,
        conf_threshold=args.conf_threshold,
    )
    detector.annotate_video(args.input_video, args.output_video)
    print(f"Wrote demo video: {args.output_video}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sprint 1 volleyball ball detector workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract frames from clips")
    extract_parser.add_argument("--manifest", default="data/raw/videos/manifest.csv")
    extract_parser.add_argument("--videos-dir", default="data/raw/videos")
    extract_parser.add_argument("--dataset-root", default="data/processed/ball_dataset")
    extract_parser.add_argument("--data-yaml", default="data/processed/ball_dataset.yaml")
    extract_parser.add_argument("--every-n-frames", type=int, default=3)
    extract_parser.add_argument("--max-frames-per-clip", type=int, default=300)
    extract_parser.set_defaults(func=cmd_extract)

    train_parser = subparsers.add_parser("train", help="Train YOLO on labeled dataset")
    train_parser.add_argument("--data-yaml", default="data/processed/ball_dataset.yaml")
    train_parser.add_argument("--base-model", default="yolov8n.pt")
    train_parser.add_argument("--imgsz", type=int, default=1280)
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--project-dir", default="models")
    train_parser.add_argument("--run-name", default="volleyball_ball_yolo")
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate model on labeled frames")
    eval_parser.add_argument("--model-path", required=True)
    eval_parser.add_argument("--images-dir", default="data/processed/ball_dataset/images/test")
    eval_parser.add_argument("--labels-dir", default="data/processed/ball_dataset/labels/test")
    eval_parser.add_argument("--class-id", type=int, default=0)
    eval_parser.add_argument("--conf-threshold", type=float, default=0.2)
    eval_parser.add_argument("--iou-threshold", type=float, default=0.3)
    eval_parser.set_defaults(func=cmd_eval)

    demo_parser = subparsers.add_parser("demo", help="Render demo video with detections")
    demo_parser.add_argument("--model-path", required=True)
    demo_parser.add_argument("--input-video", required=True)
    demo_parser.add_argument("--output-video", default="data/processed/demo_ball_detections.mp4")
    demo_parser.add_argument("--class-id", type=int, default=0)
    demo_parser.add_argument("--conf-threshold", type=float, default=0.2)
    demo_parser.set_defaults(func=cmd_demo)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
