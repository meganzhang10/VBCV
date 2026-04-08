"""Sprint 3 CLI for coordinate calibration and court-plane projection."""

from __future__ import annotations

import argparse

from src.analytics.court_calibration import (
    calibrate_and_project,
    draw_topdown_court_overlay,
    export_calibration_report,
    export_projected_track,
    write_points_template,
)


def cmd_template(args: argparse.Namespace) -> None:
    write_points_template(args.output_points)
    print(f"Wrote template point file: {args.output_points}")
    print("Fill pixel coordinates from a representative frame before running calibration.")


def cmd_run(args: argparse.Namespace) -> None:
    calibration, projected, landing, speed = calibrate_and_project(
        points_json=args.points_json,
        track_csv=args.track_csv,
    )
    export_projected_track(projected, args.output_projected_csv)
    export_calibration_report(
        calibration=calibration,
        landing_point_m=landing,
        speed_mps=speed,
        out_json=args.output_report_json,
    )
    draw_topdown_court_overlay(landing_point_m=landing, out_image=args.output_overlay_image)

    print(f"Wrote: {args.output_projected_csv}")
    print(f"Wrote: {args.output_report_json}")
    print(f"Wrote: {args.output_overlay_image}")
    if landing is not None:
        print(f"Approx landing (court m): x={landing[0]:.2f}, y={landing[1]:.2f}")
    if speed is not None:
        print(f"Approx 2D speed: {speed:.2f} m/s ({speed * 3.6:.2f} km/h)")
    print("Assumption: this is a court-plane approximation, not full 3D trajectory.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sprint 3 coordinate calibration workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser("template", help="Write calibration points template JSON")
    template_parser.add_argument("--output-points", default="data/processed/calibration_points.json")
    template_parser.set_defaults(func=cmd_template)

    run_parser = subparsers.add_parser("run", help="Run homography calibration and projection")
    run_parser.add_argument("--points-json", default="data/processed/calibration_points.json")
    run_parser.add_argument("--track-csv", default="data/processed/ball_track.csv")
    run_parser.add_argument("--output-projected-csv", default="data/processed/ball_track_court.csv")
    run_parser.add_argument("--output-report-json", default="data/processed/calibration_report.json")
    run_parser.add_argument("--output-overlay-image", default="data/processed/court_overlay_landing.png")
    run_parser.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
