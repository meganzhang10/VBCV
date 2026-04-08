"""Run the full VBCV pipeline on a single serve video.

Usage:
    python3 -m scripts.run_pipeline --input-video data/raw/videos/sample_behind_server_02.mp4

    # With calibration (after you've filled in the calibration points):
    python3 -m scripts.run_pipeline \
        --input-video data/raw/videos/sample_behind_server_02.mp4 \
        --calibration-json data/processed/calibration_points.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.detection.ball_detector import BallDetector
from src.tracking.single_ball_tracker import (
    interpolate_missing,
    render_trail_video,
    run_tracking,
    smooth_track,
    write_track_csv,
    write_track_json,
)


def run_detection_and_tracking(
    input_video: Path,
    model_path: Path,
    output_dir: Path,
    conf_threshold: float = 0.1,
    max_step_px: float = 120.0,
    max_gap: int = 5,
    smooth_window: int = 5,
    trail_length: int = 30,
) -> tuple[Path, float]:
    """Run detection + tracking and return (track_csv_path, fps)."""
    print("\n=== Step 1: Ball Detection + Tracking ===")
    detector = BallDetector(
        model_path=str(model_path),
        class_id=0,
        conf_threshold=conf_threshold,
    )

    points, fps = run_tracking(
        detector=detector,
        input_video=input_video,
        max_step_px=max_step_px,
    )

    detected = sum(1 for p in points if p.detected)
    interpolated = sum(1 for p in points if p.interpolated)
    print(f"  Frames: {len(points)}")
    print(f"  Detected: {detected} ({100 * detected / len(points):.1f}%)")

    interpolate_missing(points, max_gap=max_gap)
    interpolated_after = sum(1 for p in points if p.interpolated) - interpolated
    print(f"  Interpolated: {interpolated_after} gap frames filled")

    smooth_track(points, window_size=smooth_window)

    track_csv = output_dir / "ball_track.csv"
    track_json = output_dir / "ball_track.json"
    trail_video = output_dir / "ball_track_trail.mp4"

    write_track_csv(points, track_csv)
    write_track_json(points, track_json)
    print(f"  Track CSV: {track_csv}")
    print(f"  Track JSON: {track_json}")

    print("  Rendering trail video...")
    render_trail_video(
        input_video=input_video,
        output_video=trail_video,
        points=points,
        trail_length=trail_length,
    )
    print(f"  Trail video: {trail_video}")

    return track_csv, fps


def run_calibration(
    calibration_json: Path,
    track_csv: Path,
    output_dir: Path,
) -> Path:
    """Run court calibration + projection. Returns projected track CSV path."""
    from src.analytics.court_calibration import (
        calibrate_and_project,
        draw_topdown_court_overlay,
        export_calibration_report,
        export_projected_track,
    )

    print("\n=== Step 2: Court Calibration ===")

    calibration, projected_df, landing, speed = calibrate_and_project(
        points_json=str(calibration_json),
        track_csv=str(track_csv),
    )

    projected_csv = output_dir / "ball_track_court.csv"
    report_json = output_dir / "calibration_report.json"
    court_overlay = output_dir / "court_overlay_landing.png"

    export_projected_track(projected_df, projected_csv)
    export_calibration_report(calibration, landing, speed, report_json)
    draw_topdown_court_overlay(landing_point_m=landing, out_image=court_overlay)

    print(f"  Projected track: {projected_csv}")
    print(f"  Calibration report: {report_json}")
    print(f"  Court overlay: {court_overlay}")
    if landing:
        print(f"  Estimated landing: ({landing[0]:.2f}m, {landing[1]:.2f}m)")
    if speed:
        print(f"  Estimated speed: {speed:.1f} m/s ({speed * 3.6:.1f} km/h)")

    return projected_csv


def run_speed_estimation(
    projected_csv: Path,
    input_video: Path,
    output_dir: Path,
) -> None:
    """Run speed estimation pipeline."""
    from src.analytics.speed_estimation import (
        export_speed_csv,
        export_speed_summary_json,
        render_speed_overlay_video,
        run_speed_estimation as _run_speed,
    )

    print("\n=== Step 3: Speed Estimation ===")

    speed_df, summary = _run_speed(projected_track_csv=projected_csv)

    speed_csv = output_dir / "ball_speed.csv"
    speed_json = output_dir / "serve_speed_summary.json"
    speed_video = output_dir / "serve_speed_overlay.mp4"

    export_speed_csv(speed_df, speed_csv)
    export_speed_summary_json(summary, speed_json)
    render_speed_overlay_video(
        input_video=input_video,
        speed_df=speed_df,
        summary=summary,
        output_video=speed_video,
    )

    print(f"  Speed CSV: {speed_csv}")
    print(f"  Speed summary: {speed_json}")
    print(f"  Speed overlay video: {speed_video}")
    if summary.max_speed_kph:
        print(f"  Max speed: {summary.max_speed_kph:.1f} km/h")
    if summary.avg_speed_first_0_3s_kph:
        print(f"  Avg first 0.3s: {summary.avg_speed_first_0_3s_kph:.1f} km/h")


def run_trajectory(
    projected_csv: Path,
    input_video: Path,
    output_dir: Path,
) -> None:
    """Run trajectory arc fitting + landing detection."""
    from src.analytics.trajectory import (
        export_arc_csv,
        export_landing_json,
        render_arc_overlay_video,
        render_court_landing_map,
        run_trajectory_analysis,
    )

    print("\n=== Step 4: Trajectory + Landing ===")

    arc_df, landing = run_trajectory_analysis(projected_track_csv=projected_csv)

    arc_csv = output_dir / "ball_trajectory_arc.csv"
    landing_json = output_dir / "landing_estimate.json"
    arc_video = output_dir / "trajectory_arc_overlay.mp4"
    court_map = output_dir / "landing_court_map.png"

    export_arc_csv(arc_df, arc_csv)
    export_landing_json(landing, landing_json)
    render_arc_overlay_video(
        input_video=input_video,
        track_with_arc_df=arc_df,
        landing=landing,
        output_video=arc_video,
    )
    render_court_landing_map(landing=landing, out_image=court_map)

    print(f"  Arc CSV: {arc_csv}")
    print(f"  Landing estimate: {landing_json}")
    print(f"  Arc overlay video: {arc_video}")
    print(f"  Court landing map: {court_map}")
    if landing:
        print(f"  Landing point: ({landing.court_x_m}, {landing.court_y_m}) m")
        print(f"  Method: {landing.method}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full VBCV pipeline on a serve video")
    parser.add_argument("--input-video", required=True, help="Path to serve video (.mp4)")
    parser.add_argument(
        "--model-path",
        default="models/volleyball_ball_yolo/weights/best.pt",
        help="Path to YOLO weights",
    )
    parser.add_argument(
        "--calibration-json",
        default=None,
        help="Path to calibration points JSON (skip calibration if not provided)",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (auto-generated if not set)")
    parser.add_argument("--conf-threshold", type=float, default=0.1, help="Detection confidence threshold")
    args = parser.parse_args()

    input_video = Path(args.input_video)
    if not input_video.exists():
        print(f"ERROR: Video not found: {input_video}")
        return

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Run 'python3 -m scripts.setup_and_train' first to train the model.")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("data/processed/runs") / input_video.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input video:  {input_video}")
    print(f"Model:        {model_path}")
    print(f"Output dir:   {output_dir}")

    # Step 1: Always run detection + tracking
    track_csv, fps = run_detection_and_tracking(
        input_video=input_video,
        model_path=model_path,
        output_dir=output_dir,
        conf_threshold=args.conf_threshold,
    )

    # Step 2-4: Only if calibration is provided
    if args.calibration_json:
        cal_path = Path(args.calibration_json)
        if not cal_path.exists():
            print(f"\nERROR: Calibration JSON not found: {cal_path}")
            return

        projected_csv = run_calibration(cal_path, track_csv, output_dir)
        run_speed_estimation(projected_csv, input_video, output_dir)
        run_trajectory(projected_csv, input_video, output_dir)
    else:
        print("\n=== Calibration Skipped ===")
        print("  No --calibration-json provided.")
        print("  To enable speed/trajectory analysis, create calibration points:")
        print(f"    python3 -m src.analytics.sprint3 template --output-points data/processed/calibration_points.json")
        print(f"  Then edit the JSON with pixel coordinates from your video and re-run:")
        print(f"    python3 -m scripts.run_pipeline --input-video {input_video} --calibration-json data/processed/calibration_points.json")

    print("\n=== Done ===")
    print(f"All outputs in: {output_dir}")
    print(f"\nFiles generated:")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
