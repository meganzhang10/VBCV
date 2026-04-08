"""Sprint 4 CLI for serve speed estimation."""

from __future__ import annotations

import argparse

from src.analytics.speed_estimation import (
    export_speed_csv,
    export_speed_summary_json,
    render_speed_overlay_video,
    run_speed_estimation,
)


def cmd_run(args: argparse.Namespace) -> None:
    speed_df, summary = run_speed_estimation(
        projected_track_csv=args.projected_track_csv,
        smooth_window=args.smooth_window,
        early_window_sec=args.early_window_sec,
    )
    export_speed_csv(speed_df, args.output_speed_csv)
    export_speed_summary_json(summary, args.output_summary_json)
    render_speed_overlay_video(
        input_video=args.input_video,
        speed_df=speed_df,
        summary=summary,
        output_video=args.output_video,
    )
    print(f"Wrote: {args.output_speed_csv}")
    print(f"Wrote: {args.output_summary_json}")
    print(f"Wrote: {args.output_video}")
    if summary.max_speed_mps is not None:
        print(f"Max speed: {summary.max_speed_mps:.2f} m/s ({summary.max_speed_kph:.2f} km/h)")
    if summary.avg_speed_first_0_3s_mps is not None:
        print(
            "Avg first 0.3s: "
            f"{summary.avg_speed_first_0_3s_mps:.2f} m/s "
            f"({summary.avg_speed_first_0_3s_kph:.2f} km/h)"
        )
    print(summary.assumption)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sprint 4 speed estimation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Estimate serve speed and render overlay")
    run_parser.add_argument("--projected-track-csv", default="data/processed/ball_track_court.csv")
    run_parser.add_argument("--input-video", required=True)
    run_parser.add_argument("--output-speed-csv", default="data/processed/ball_speed.csv")
    run_parser.add_argument("--output-summary-json", default="data/processed/serve_speed_summary.json")
    run_parser.add_argument("--output-video", default="data/processed/serve_speed_overlay.mp4")
    run_parser.add_argument("--smooth-window", type=int, default=5)
    run_parser.add_argument("--early-window-sec", type=float, default=0.3)
    run_parser.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
