"""Polished Streamlit product demo for serve analytics (Sprint 7)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd
import streamlit as st

from src.analytics.court_calibration import (
    calibrate_and_project,
    draw_topdown_court_overlay,
    export_calibration_report,
    export_projected_track,
)
from src.analytics.session_consistency import SessionSummary, compute_session_from_manifest
from src.analytics.speed_estimation import (
    export_speed_csv,
    export_speed_summary_json,
    render_speed_overlay_video,
    run_speed_estimation,
)
from src.analytics.target_zone_coaching import (
    TargetCoachingSummary,
    score_serves_for_target_zone,
    summarize_target_coaching,
)
from src.analytics.trajectory import (
    CourtRegion,
    export_arc_csv,
    export_landing_json,
    render_arc_overlay_video,
    render_court_landing_map,
    run_trajectory_analysis,
)
from src.detection.ball_detector import BallDetector
from src.tracking.single_ball_tracker import (
    interpolate_missing,
    render_trail_video,
    run_tracking,
    smooth_track,
    write_track_csv,
    write_track_json,
)
from src.visualization.session_store import (
    SessionServeRecord,
    add_serve_record,
    build_session_manifest_path,
    get_session_serves,
    list_session_ids,
    load_session_store,
    sanitize_slug,
    save_session_store,
    utc_now_iso,
    write_session_manifest,
)


def _fmt(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{precision}f}"


def _read_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def _build_heatmap_data(serves_df: pd.DataFrame, bin_size_m: float = 0.75) -> pd.DataFrame:
    data = serves_df.dropna(subset=["landing_x_m", "landing_y_m"]).copy()
    if data.empty:
        return pd.DataFrame(columns=["x_bin", "y_bin", "count"])

    data["x_bin"] = (data["landing_x_m"] / bin_size_m).round().astype(int) * bin_size_m
    data["y_bin"] = (data["landing_y_m"] / bin_size_m).round().astype(int) * bin_size_m
    return (
        data.groupby(["x_bin", "y_bin"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )


def _show_summary_cards(summary: SessionSummary) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Serves", f"{summary.serve_count}")
    col2.metric("Avg Speed (km/h)", _fmt(summary.average_speed_kph, precision=1))
    col3.metric("Speed Variance", _fmt(summary.speed_variance_kph2, precision=2))
    col4.metric("Consistency Score", _fmt(summary.consistency_score, precision=1))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("In %", _fmt(summary.in_percentage, precision=1))
    col6.metric("Target Accuracy %", _fmt(summary.target_zone_accuracy, precision=1))
    col7.metric("Landing Spread (m)", _fmt(summary.landing_zone_spread_m, precision=2))
    col8.metric("Depth Consistency (m)", _fmt(summary.depth_consistency_m, precision=2))


def _render_session_visuals(serves_df: pd.DataFrame) -> None:
    st.subheader("Consistency Plots")

    heatmap_df = _build_heatmap_data(serves_df)
    st.vega_lite_chart(
        heatmap_df,
        {
            "mark": "rect",
            "encoding": {
                "x": {"field": "x_bin", "type": "quantitative", "title": "Court X (m)"},
                "y": {"field": "y_bin", "type": "quantitative", "title": "Court Y (m)"},
                "color": {"field": "count", "type": "quantitative", "title": "Serve Count"},
                "tooltip": ["x_bin", "y_bin", "count"],
            },
        },
        use_container_width=True,
    )

    scatter_df = serves_df.dropna(subset=["landing_x_m", "landing_y_m"]).copy()
    if not scatter_df.empty:
        scatter_df["result"] = scatter_df["in_bounds"].map({True: "in", False: "out"}).fillna("unknown")
    st.vega_lite_chart(
        scatter_df,
        {
            "mark": {"type": "point", "filled": True, "size": 90},
            "encoding": {
                "x": {"field": "landing_x_m", "type": "quantitative", "title": "Court X (m)"},
                "y": {"field": "landing_y_m", "type": "quantitative", "title": "Court Y (m)"},
                "color": {"field": "result", "type": "nominal", "title": "Result"},
                "tooltip": ["serve_id", "speed_kph", "landing_x_m", "landing_y_m", "in_bounds", "target_hit"],
            },
        },
        use_container_width=True,
    )

    speed_df = serves_df.dropna(subset=["speed_kph"])
    st.vega_lite_chart(
        speed_df,
        {
            "mark": "bar",
            "encoding": {
                "x": {"bin": True, "field": "speed_kph", "type": "quantitative", "title": "Speed (km/h)"},
                "y": {"aggregate": "count", "type": "quantitative", "title": "Serve Count"},
            },
        },
        use_container_width=True,
    )

    st.dataframe(serves_df, use_container_width=True)


def _render_target_zone_coaching(scored_df: pd.DataFrame, summary: TargetCoachingSummary) -> None:
    st.subheader("Target-Zone Coaching")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scored Serves", f"{summary.scored_serves}/{summary.serve_count}")
    c2.metric("Avg Miss Dist (m)", _fmt(summary.avg_distance_to_target_m, precision=2))
    c3.metric("Best Dist (m)", _fmt(summary.best_distance_to_target_m, precision=2))
    c4.metric("Bullseye %", _fmt(summary.bullseye_rate_pct, precision=1))

    c5 = st.columns(1)[0]
    c5.metric("Avg Target Score", _fmt(summary.average_target_proximity_score, precision=1))
    st.caption(summary.coaching_note)

    ranking_df = scored_df.dropna(subset=["target_proximity_score"]).copy()
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values("target_proximity_score", ascending=False)
        st.vega_lite_chart(
            ranking_df,
            {
                "mark": {"type": "bar", "cornerRadiusEnd": 3},
                "encoding": {
                    "x": {"field": "serve_id", "type": "nominal", "sort": "-y", "title": "Serve"},
                    "y": {
                        "field": "target_proximity_score",
                        "type": "quantitative",
                        "title": "Target Proximity Score",
                    },
                    "tooltip": [
                        "serve_id",
                        "distance_to_target_m",
                        "target_proximity_score",
                        "bullseye_hit",
                    ],
                },
            },
            use_container_width=True,
        )
    st.dataframe(
        scored_df[
            [
                "serve_id",
                "speed_kph",
                "landing_x_m",
                "landing_y_m",
                "distance_to_target_m",
                "target_proximity_score",
                "bullseye_hit",
            ]
        ],
        use_container_width=True,
    )


def _process_uploaded_serve(
    *,
    uploaded_video_name: str,
    uploaded_video_bytes: bytes,
    session_id: str,
    serve_label: str,
    model_path: str,
    calibration_points_json: str,
    conf_threshold: float,
    max_step_px: float,
    max_gap: int,
    smooth_window: int,
    trail_length: int,
    early_window_sec: float,
    arc_degree: int,
) -> SessionServeRecord:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_session = sanitize_slug(session_id)
    safe_serve = sanitize_slug(serve_label or Path(uploaded_video_name).stem)
    serve_id = f"{safe_serve}_{timestamp}"

    out_dir = Path("data/processed/sessions") / safe_session / serve_id
    out_dir.mkdir(parents=True, exist_ok=True)

    input_video = out_dir / f"input_{Path(uploaded_video_name).name}"
    input_video.write_bytes(uploaded_video_bytes)

    detector = BallDetector(model_path=model_path, class_id=0, conf_threshold=conf_threshold)
    points, _fps = run_tracking(detector=detector, input_video=input_video, max_step_px=max_step_px)
    interpolate_missing(points, max_gap=max_gap)
    smooth_track(points, window_size=smooth_window)

    track_csv = out_dir / "ball_track.csv"
    track_json = out_dir / "ball_track.json"
    trail_video = out_dir / "ball_track_trail.mp4"
    write_track_csv(points, track_csv)
    write_track_json(points, track_json)
    render_trail_video(input_video=input_video, output_video=trail_video, points=points, trail_length=trail_length)

    calibration, projected_df, landing0, speed0 = calibrate_and_project(
        points_json=calibration_points_json,
        track_csv=track_csv,
    )
    projected_track_csv = out_dir / "ball_track_court.csv"
    calibration_report = out_dir / "calibration_report.json"
    initial_court_map = out_dir / "court_overlay_landing.png"
    export_projected_track(projected_df, projected_track_csv)
    export_calibration_report(calibration, landing0, speed0, calibration_report)
    draw_topdown_court_overlay(landing_point_m=landing0, out_image=initial_court_map)

    speed_df, speed_summary = run_speed_estimation(projected_track_csv=projected_track_csv, early_window_sec=early_window_sec)
    speed_csv = out_dir / "ball_speed.csv"
    speed_summary_json = out_dir / "serve_speed_summary.json"
    speed_video = out_dir / "serve_speed_overlay.mp4"
    export_speed_csv(speed_df, speed_csv)
    export_speed_summary_json(speed_summary, speed_summary_json)
    render_speed_overlay_video(input_video=input_video, speed_df=speed_df, summary=speed_summary, output_video=speed_video)

    arc_df, landing = run_trajectory_analysis(projected_track_csv=projected_track_csv, degree=arc_degree)
    arc_csv = out_dir / "ball_trajectory_arc.csv"
    landing_json = out_dir / "landing_estimate.json"
    trajectory_video = out_dir / "trajectory_arc_overlay.mp4"
    court_map = out_dir / "landing_court_map.png"
    export_arc_csv(arc_df, arc_csv)
    export_landing_json(landing, landing_json)
    render_arc_overlay_video(input_video=input_video, track_with_arc_df=arc_df, landing=landing, output_video=trajectory_video)
    render_court_landing_map(landing=landing, out_image=court_map)

    landing_x = None if landing is None else landing.court_x_m
    landing_y = None if landing is None else landing.court_y_m

    return SessionServeRecord(
        serve_id=serve_id,
        created_at=utc_now_iso(),
        input_video=str(input_video),
        output_dir=str(out_dir),
        projected_track_csv=str(projected_track_csv),
        trail_video=str(trail_video),
        speed_overlay_video=str(speed_video),
        trajectory_overlay_video=str(trajectory_video),
        court_map_image=str(court_map),
        speed_kph=speed_summary.max_speed_kph,
        landing_x_m=landing_x,
        landing_y_m=landing_y,
    )


def _build_export_zip(serve: dict[str, object]) -> Path:
    source_dir = Path(str(serve["output_dir"]))
    zip_base = source_dir / "annotated_exports"
    archive_path = shutil.make_archive(base_name=str(zip_base), format="zip", root_dir=str(source_dir))
    return Path(archive_path)


def _to_optional_float(val: object) -> float | None:
    if val is None:
        return None
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _serve_card(serve: dict[str, object], column: st.delta_generator.DeltaGenerator) -> None:  # type: ignore[name-defined]
    column.markdown(f"**{serve['serve_id']}**")
    column.caption(f"Speed: {_fmt(_to_optional_float(serve.get('speed_kph')), precision=1)} km/h")
    column.caption(
        f"Landing: ({_fmt(_to_optional_float(serve.get('landing_x_m')), precision=2)}, "
        f"{_fmt(_to_optional_float(serve.get('landing_y_m')), precision=2)}) m"
    )
    trajectory_video = Path(str(serve["trajectory_overlay_video"]))
    if trajectory_video.exists():
        column.video(_read_bytes(trajectory_video))


def build_dashboard() -> None:
    st.set_page_config(page_title="Serve Product Demo", layout="wide")
    st.title("Sprint 7 - Polished Serve Analytics Product")
    st.caption("Upload a serve, auto-process it, save to session, compare serves, and export annotated clips.")

    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #f7f9fb 0%, #ecf2f8 100%); }
        .block-container { padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    store = load_session_store()
    existing_sessions = list_session_ids(store)
    default_session = existing_sessions[0] if existing_sessions else "demo-session"

    tab_upload, tab_session, tab_compare = st.tabs(["Upload & Process", "Session Analytics", "Compare Serves"])

    with tab_upload:
        st.subheader("Auto-process a serve clip")
        c1, c2 = st.columns([2, 1])
        session_id = c1.text_input("Session name", value=default_session)
        serve_label = c2.text_input("Serve label", value="serve")

        c3, c4 = st.columns(2)
        model_path = c3.text_input("YOLO model path", value="models/volleyball_ball_yolo/weights/best.pt")
        calibration_points_json = c4.text_input("Calibration points JSON", value="data/processed/calibration_points.json")

        with st.expander("Advanced settings"):
            a1, a2, a3, a4 = st.columns(4)
            conf_threshold = a1.number_input("conf_threshold", value=0.2, min_value=0.01, max_value=0.99)
            max_step_px = a2.number_input("max_step_px", value=120.0, min_value=10.0)
            max_gap = int(a3.number_input("max_gap", value=5, min_value=0, step=1))
            smooth_window = int(a4.number_input("smooth_window", value=5, min_value=1, step=1))
            b1, b2, b3 = st.columns(3)
            trail_length = int(b1.number_input("trail_length", value=30, min_value=1, step=1))
            early_window_sec = b2.number_input("early_window_sec", value=0.3, min_value=0.05)
            arc_degree = int(b3.number_input("arc_degree", value=2, min_value=1, max_value=5, step=1))

        uploaded = st.file_uploader("Upload serve clip (.mp4)", type=["mp4", "mov", "m4v"])

        if st.button("Process Serve", type="primary"):
            if uploaded is None:
                st.error("Upload a clip first.")
            elif not Path(model_path).exists():
                st.error(f"Model not found: {model_path}")
            elif not Path(calibration_points_json).exists():
                st.error(f"Calibration JSON not found: {calibration_points_json}")
            else:
                with st.spinner("Running detection, tracking, calibration, speed, and trajectory analysis..."):
                    try:
                        record = _process_uploaded_serve(
                            uploaded_video_name=uploaded.name,
                            uploaded_video_bytes=uploaded.getvalue(),
                            session_id=session_id,
                            serve_label=serve_label,
                            model_path=model_path,
                            calibration_points_json=calibration_points_json,
                            conf_threshold=float(conf_threshold),
                            max_step_px=float(max_step_px),
                            max_gap=max_gap,
                            smooth_window=smooth_window,
                            trail_length=trail_length,
                            early_window_sec=float(early_window_sec),
                            arc_degree=arc_degree,
                        )
                    except Exception as exc:  # pragma: no cover - UI path
                        st.exception(exc)
                    else:
                        add_serve_record(store, sanitize_slug(session_id), record)
                        save_session_store(store)
                        st.success(f"Processed and saved serve: {record.serve_id}")

                        st.subheader("Annotated output")
                        st.video(_read_bytes(record.trajectory_overlay_video))
                        st.write(
                            f"Speed: {_fmt(record.speed_kph, precision=1)} km/h | "
                            f"Landing: ({_fmt(record.landing_x_m, precision=2)}, {_fmt(record.landing_y_m, precision=2)}) m"
                        )

                        zip_path = _build_export_zip(asdict(record))
                        st.download_button(
                            "Download annotated bundle (.zip)",
                            data=_read_bytes(zip_path),
                            file_name=zip_path.name,
                            mime="application/zip",
                        )

    with tab_session:
        st.subheader("Session summary stats")
        sessions = list_session_ids(store)
        if not sessions:
            st.info("No saved sessions yet. Process at least one serve in Upload & Process.")
        else:
            selected_session = st.selectbox("Choose session", options=sessions)
            c1, c2, c3, c4 = st.columns(4)
            x_min = c1.number_input("target x_min_m", value=9.0)
            x_max = c2.number_input("target x_max_m", value=18.0)
            y_min = c3.number_input("target y_min_m", value=0.0)
            y_max = c4.number_input("target y_max_m", value=9.0)
            target_region = CourtRegion(float(x_min), float(x_max), float(y_min), float(y_max))

            manifest_path = build_session_manifest_path(selected_session)
            manifest_df = write_session_manifest(store, selected_session, manifest_path)
            if manifest_df.empty:
                st.warning("Selected session has no valid serves yet.")
            else:
                serves_df, summary = compute_session_from_manifest(manifest_path, target_region=target_region)
                _show_summary_cards(summary)
                _render_session_visuals(serves_df)
                scored_df = score_serves_for_target_zone(serves_df, target_region=target_region)
                coaching_summary = summarize_target_coaching(scored_df)
                _render_target_zone_coaching(scored_df, coaching_summary)

    with tab_compare:
        st.subheader("Side-by-side serve comparison")
        sessions = list_session_ids(store)
        if not sessions:
            st.info("No serves available for comparison yet.")
        else:
            selected_session = st.selectbox("Session", options=sessions, key="compare_session")
            serves = get_session_serves(store, selected_session)
            if len(serves) < 2:
                st.warning("Need at least two serves in this session.")
            else:
                serve_ids = [str(s["serve_id"]) for s in serves]
                left_id = st.selectbox("Serve A", options=serve_ids, index=0)
                right_id = st.selectbox("Serve B", options=serve_ids, index=1 if len(serve_ids) > 1 else 0)

                left_serve = next(s for s in serves if s["serve_id"] == left_id)
                right_serve = next(s for s in serves if s["serve_id"] == right_id)

                col_a, col_b = st.columns(2)
                _serve_card(left_serve, col_a)
                _serve_card(right_serve, col_b)

                comp_df = pd.DataFrame(
                    [
                        {
                            "metric": "speed_kph",
                            "serve_a": left_serve.get("speed_kph"),
                            "serve_b": right_serve.get("speed_kph"),
                        },
                        {
                            "metric": "landing_x_m",
                            "serve_a": left_serve.get("landing_x_m"),
                            "serve_b": right_serve.get("landing_x_m"),
                        },
                        {
                            "metric": "landing_y_m",
                            "serve_a": left_serve.get("landing_y_m"),
                            "serve_b": right_serve.get("landing_y_m"),
                        },
                    ]
                )
                st.dataframe(comp_df, use_container_width=True)

                z1, z2 = st.columns(2)
                left_zip = _build_export_zip(left_serve)
                right_zip = _build_export_zip(right_serve)
                z1.download_button(
                    "Download Serve A exports",
                    data=_read_bytes(left_zip),
                    file_name=left_zip.name,
                    mime="application/zip",
                )
                z2.download_button(
                    "Download Serve B exports",
                    data=_read_bytes(right_zip),
                    file_name=right_zip.name,
                    mime="application/zip",
                )


if __name__ == "__main__":
    build_dashboard()
