"""Sprint 2 CLI for single-ball tracking and trail rendering."""

from __future__ import annotations

import argparse

from src.detection.ball_detector import BallDetector
from src.tracking.single_ball_tracker import (
    interpolate_missing,
    render_trail_video,
    run_tracking,
    smooth_track,
    write_track_csv,
    write_track_json,
)


def cmd_run(args: argparse.Namespace) -> None:
    detector = BallDetector(
        model_path=args.model_path,
        class_id=args.class_id,
        conf_threshold=args.conf_threshold,
    )
    points, fps = run_tracking(
        detector=detector,
        input_video=args.input_video,
        max_step_px=args.max_step_px,
    )
    interpolate_missing(points, max_gap=args.max_gap)
    smooth_track(points, window_size=args.smooth_window)
    write_track_csv(points, args.output_csv)
    write_track_json(points, args.output_json)
    render_trail_video(
        input_video=args.input_video,
        output_video=args.output_video,
        points=points,
        trail_length=args.trail_length,
    )

    detected = sum(1 for p in points if p.detected)
    interpolated = sum(1 for p in points if p.interpolated)
    visible = sum(1 for p in points if p.raw_x is not None and p.raw_y is not None)
    total = len(points)
    print(
        f"tracked_frames={total} fps={fps:.2f} detected={detected} interpolated={interpolated} "
        f"visible_after_fill={visible}"
    )
    print(f"Wrote: {args.output_video}")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sprint 2 ball tracking workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Track ball, smooth path, and export outputs")
    run_parser.add_argument("--model-path", required=True)
    run_parser.add_argument("--input-video", required=True)
    run_parser.add_argument("--output-video", default="data/processed/ball_track_trail.mp4")
    run_parser.add_argument("--output-csv", default="data/processed/ball_track.csv")
    run_parser.add_argument("--output-json", default="data/processed/ball_track.json")
    run_parser.add_argument("--class-id", type=int, default=0)
    run_parser.add_argument("--conf-threshold", type=float, default=0.2)
    run_parser.add_argument("--max-step-px", type=float, default=120.0)
    run_parser.add_argument("--max-gap", type=int, default=5)
    run_parser.add_argument("--smooth-window", type=int, default=5)
    run_parser.add_argument("--trail-length", type=int, default=30)
    run_parser.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
