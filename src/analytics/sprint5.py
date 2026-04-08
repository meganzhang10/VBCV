"""Sprint 5 CLI for trajectory arc overlay and landing detection."""

from __future__ import annotations

import argparse

from src.analytics.trajectory import (
    export_arc_csv,
    export_landing_json,
    render_arc_overlay_video,
    render_court_landing_map,
    run_trajectory_analysis,
)


def cmd_run(args: argparse.Namespace) -> None:
    arc_df, landing = run_trajectory_analysis(
        projected_track_csv=args.projected_track_csv,
        degree=args.arc_degree,
    )
    export_arc_csv(arc_df, args.output_arc_csv)
    export_landing_json(landing, args.output_landing_json)
    render_arc_overlay_video(
        input_video=args.input_video,
        track_with_arc_df=arc_df,
        landing=landing,
        output_video=args.output_video,
    )
    render_court_landing_map(landing=landing, out_image=args.output_court_map)

    print(f"Wrote: {args.output_arc_csv}")
    print(f"Wrote: {args.output_landing_json}")
    print(f"Wrote: {args.output_video}")
    print(f"Wrote: {args.output_court_map}")
    if landing is not None:
        print(
            "Landing estimate: "
            f"frame={landing.frame_index} t={landing.timestamp_sec:.3f}s "
            f"pixel=({landing.pixel_x:.1f},{landing.pixel_y:.1f}) "
            f"court=({landing.court_x_m},{landing.court_y_m}) method={landing.method}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sprint 5 trajectory arc and landing workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Fit arc, detect landing, and render overlays")
    run_parser.add_argument("--projected-track-csv", default="data/processed/ball_track_court.csv")
    run_parser.add_argument("--input-video", required=True)
    run_parser.add_argument("--output-arc-csv", default="data/processed/ball_trajectory_arc.csv")
    run_parser.add_argument("--output-landing-json", default="data/processed/landing_estimate.json")
    run_parser.add_argument("--output-video", default="data/processed/trajectory_arc_overlay.mp4")
    run_parser.add_argument("--output-court-map", default="data/processed/landing_court_map.png")
    run_parser.add_argument("--arc-degree", type=int, default=2)
    run_parser.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
