from pathlib import Path

from src.visualization.session_store import (
    SessionServeRecord,
    add_serve_record,
    build_session_manifest_path,
    get_session_serves,
    list_session_ids,
    load_session_store,
    sanitize_slug,
    save_session_store,
    write_session_manifest,
)


def test_sanitize_slug_normalizes_strings() -> None:
    assert sanitize_slug(" Team Session #1 ") == "Team-Session-1"
    assert sanitize_slug("***") == "session"


def test_store_roundtrip_and_manifest(tmp_path: Path) -> None:
    store_path = tmp_path / "store.json"

    store = load_session_store(store_path)
    record = SessionServeRecord(
        serve_id="serve_001",
        created_at="2026-04-06T00:00:00+00:00",
        input_video="/tmp/input.mp4",
        output_dir="/tmp/out",
        projected_track_csv="/tmp/out/ball_track_court.csv",
        trail_video="/tmp/out/trail.mp4",
        speed_overlay_video="/tmp/out/speed.mp4",
        trajectory_overlay_video="/tmp/out/arc.mp4",
        court_map_image="/tmp/out/map.png",
        speed_kph=58.2,
        landing_x_m=11.3,
        landing_y_m=4.2,
    )

    add_serve_record(store, "session-a", record)
    save_session_store(store, store_path)

    loaded = load_session_store(store_path)
    assert list_session_ids(loaded) == ["session-a"]
    serves = get_session_serves(loaded, "session-a")
    assert len(serves) == 1
    assert serves[0]["serve_id"] == "serve_001"

    manifest_path = tmp_path / "session_manifest.csv"
    manifest_df = write_session_manifest(loaded, "session-a", manifest_path)
    assert len(manifest_df) == 1
    assert manifest_df.iloc[0]["projected_track_csv"] == "/tmp/out/ball_track_court.csv"


def test_build_session_manifest_path() -> None:
    path = build_session_manifest_path("demo")
    assert path.as_posix().endswith("data/processed/sessions/demo/session_serves.csv")
