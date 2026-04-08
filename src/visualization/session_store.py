"""Persistence helpers for saved serve sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

DEFAULT_STORE_PATH = Path("data/processed/serve_sessions.json")


@dataclass(frozen=True)
class SessionServeRecord:
    serve_id: str
    created_at: str
    input_video: str
    output_dir: str
    projected_track_csv: str
    trail_video: str
    speed_overlay_video: str
    trajectory_overlay_video: str
    court_map_image: str
    speed_kph: float | None
    landing_x_m: float | None
    landing_y_m: float | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_slug(raw: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw.strip())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "session"


def load_session_store(path: str | Path = DEFAULT_STORE_PATH) -> dict[str, Any]:
    store_path = Path(path)
    if not store_path.exists():
        return {"sessions": []}
    payload = json.loads(store_path.read_text(encoding="utf-8"))
    if "sessions" not in payload or not isinstance(payload["sessions"], list):
        raise ValueError("Session store is invalid: expected top-level 'sessions' list")
    return payload


def save_session_store(store: dict[str, Any], path: str | Path = DEFAULT_STORE_PATH) -> None:
    store_path = Path(path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _ensure_session(store: dict[str, Any], session_id: str) -> dict[str, Any]:
    for session in store["sessions"]:
        if session.get("session_id") == session_id:
            if "serves" not in session:
                session["serves"] = []
            return session
    session = {
        "session_id": session_id,
        "created_at": utc_now_iso(),
        "serves": [],
    }
    store["sessions"].append(session)
    return session


def list_session_ids(store: dict[str, Any]) -> list[str]:
    return [str(session.get("session_id")) for session in store.get("sessions", []) if session.get("session_id")]


def get_session_serves(store: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
    for session in store.get("sessions", []):
        if session.get("session_id") == session_id:
            serves = session.get("serves", [])
            return serves if isinstance(serves, list) else []
    return []


def add_serve_record(
    store: dict[str, Any],
    session_id: str,
    record: SessionServeRecord,
) -> None:
    session = _ensure_session(store, session_id)
    serves = session["serves"]
    payload = {
        "serve_id": record.serve_id,
        "created_at": record.created_at,
        "input_video": record.input_video,
        "output_dir": record.output_dir,
        "projected_track_csv": record.projected_track_csv,
        "trail_video": record.trail_video,
        "speed_overlay_video": record.speed_overlay_video,
        "trajectory_overlay_video": record.trajectory_overlay_video,
        "court_map_image": record.court_map_image,
        "speed_kph": record.speed_kph,
        "landing_x_m": record.landing_x_m,
        "landing_y_m": record.landing_y_m,
    }

    for idx, existing in enumerate(serves):
        if existing.get("serve_id") == record.serve_id:
            serves[idx] = payload
            return
    serves.append(payload)


def write_session_manifest(
    store: dict[str, Any],
    session_id: str,
    out_csv: str | Path,
) -> pd.DataFrame:
    serves = get_session_serves(store, session_id)
    manifest_df = pd.DataFrame(
        [
            {
                "serve_id": str(serve["serve_id"]),
                "projected_track_csv": str(serve["projected_track_csv"]),
            }
            for serve in serves
            if serve.get("serve_id") and serve.get("projected_track_csv")
        ]
    )
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out_path, index=False)
    return manifest_df


def build_session_manifest_path(session_id: str) -> Path:
    return Path("data/processed/sessions") / session_id / "session_serves.csv"
