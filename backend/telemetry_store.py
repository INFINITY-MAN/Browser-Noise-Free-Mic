"""SQLite + JSONL Telemetry & Opt-In Feedback Store for NEON Browser Noise-Free Mic.

Stores anonymized runtime telemetry in:
  - SQLite: monitoring/telemetry.db (tables: telemetry_events, feedback_events)
  - JSONL:  monitoring/telemetry.jsonl
  - Opt-in WAV clips (only when user explicitly opts in): monitoring/opt_in_clips/<uuid>.wav
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_RATINGS = frozenset({"clean", "over_suppressed", "residual_noise"})

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Compute linear-interpolated percentile (0..100) from a sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


class TelemetryStore:
    """Thread-safe SQLite + JSONL telemetry and feedback persistence layer."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else _DEFAULT_ROOT
        self.monitoring_dir = self.base_dir / "monitoring"
        self.db_path = self.monitoring_dir / "telemetry.db"
        self.jsonl_path = self.monitoring_dir / "telemetry.jsonl"
        self.opt_in_clips_dir = self.monitoring_dir / "opt_in_clips"
        self._lock = threading.Lock()
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.opt_in_clips_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    avg_infer_ms REAL NOT NULL,
                    rtf REAL NOT NULL,
                    est_snr_gain_db REAL NOT NULL,
                    dry_wet_pct REAL NOT NULL,
                    underruns INTEGER NOT NULL,
                    heap_mb REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    rating TEXT NOT NULL CHECK(rating IN ('clean', 'over_suppressed', 'residual_noise')),
                    dry_wet_pct REAL NOT NULL,
                    opt_in_clip_path TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_created_at ON telemetry_events(created_at);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback_events(created_at);"
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def save_opt_in_clip(self, audio_bytes: bytes) -> str:
        """Persist an opt-in WAV audio clip to monitoring/opt_in_clips/<uuid>.wav.

        Only call this when the user has explicitly opted in to sharing audio.
        Returns the POSIX-style relative path under the project root.
        """
        if not audio_bytes or not isinstance(audio_bytes, (bytes, bytearray)):
            raise ValueError("audio_bytes must be non-empty bytes.")
        self.opt_in_clips_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.wav"
        dest = self.opt_in_clips_dir / filename
        with self._lock:
            dest.write_bytes(bytes(audio_bytes))
        return f"monitoring/opt_in_clips/{filename}"

    def log_telemetry(
        self,
        *,
        model_version: str = "gtcrn_int8_v1",
        backend: str = "wasm",
        mode: str = "A",
        avg_infer_ms: float = 0.0,
        rtf: float = 0.0,
        est_snr_gain_db: float = 0.0,
        dry_wet_pct: float = 100.0,
        underruns: int = 0,
        heap_mb: float = 0.0,
        created_at: Optional[str] = None,
    ) -> int:
        """Insert a telemetry event into SQLite and append it to telemetry.jsonl."""
        ts = created_at or _utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO telemetry_events (
                        created_at, model_version, backend, mode,
                        avg_infer_ms, rtf, est_snr_gain_db, dry_wet_pct,
                        underruns, heap_mb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        str(model_version),
                        str(backend),
                        str(mode),
                        float(avg_infer_ms),
                        float(rtf),
                        float(est_snr_gain_db),
                        float(dry_wet_pct),
                        int(underruns),
                        float(heap_mb),
                    ),
                )
                row_id = int(cur.lastrowid)
                conn.commit()
            self._append_jsonl(
                {
                    "event_type": "telemetry",
                    "id": row_id,
                    "created_at": ts,
                    "model_version": str(model_version),
                    "backend": str(backend),
                    "mode": str(mode),
                    "avg_infer_ms": float(avg_infer_ms),
                    "rtf": float(rtf),
                    "est_snr_gain_db": float(est_snr_gain_db),
                    "dry_wet_pct": float(dry_wet_pct),
                    "underruns": int(underruns),
                    "heap_mb": float(heap_mb),
                }
            )
        return row_id

    def log_feedback(
        self,
        *,
        rating: str,
        model_version: str = "gtcrn_int8_v1",
        dry_wet_pct: float = 100.0,
        opt_in_clip_path: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> int:
        """Insert a user quality feedback event into SQLite and telemetry.jsonl."""
        rating_norm = str(rating).strip().lower()
        if rating_norm not in VALID_RATINGS:
            raise ValueError(
                f"Invalid rating '{rating}'. Expected one of {sorted(VALID_RATINGS)}."
            )
        ts = created_at or _utc_now_iso()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO feedback_events (
                        created_at, model_version, rating, dry_wet_pct, opt_in_clip_path
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ts,
                        str(model_version),
                        rating_norm,
                        float(dry_wet_pct),
                        opt_in_clip_path,
                    ),
                )
                row_id = int(cur.lastrowid)
                conn.commit()
            self._append_jsonl(
                {
                    "event_type": "feedback",
                    "id": row_id,
                    "created_at": ts,
                    "model_version": str(model_version),
                    "rating": rating_norm,
                    "dry_wet_pct": float(dry_wet_pct),
                    "opt_in_clip_path": opt_in_clip_path,
                }
            )
        return row_id

    def get_metrics_dashboard(self, recent_limit: int = 20) -> Dict[str, Any]:
        """Return aggregated runtime telemetry and user feedback statistics."""
        with self._lock:
            with self._connect() as conn:
                t_rows = conn.execute(
                    "SELECT backend, rtf, dry_wet_pct FROM telemetry_events"
                ).fetchall()
                f_total = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM feedback_events"
                ).fetchone()["cnt"]
                f_clean = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM feedback_events WHERE rating = 'clean'"
                ).fetchone()["cnt"]
                opt_in_count = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt FROM feedback_events
                    WHERE opt_in_clip_path IS NOT NULL AND opt_in_clip_path != ''
                    """
                ).fetchone()["cnt"]
                recent_rows = conn.execute(
                    """
                    SELECT id, created_at, model_version, rating, dry_wet_pct, opt_in_clip_path
                    FROM feedback_events
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (int(recent_limit),),
                ).fetchall()

        total_sessions = len(t_rows)
        webgpu_count = sum(
            1 for r in t_rows if str(r["backend"]).strip().lower() == "webgpu"
        )
        wasm_count = sum(
            1 for r in t_rows if str(r["backend"]).strip().lower() == "wasm"
        )
        rtfs = sorted(float(r["rtf"]) for r in t_rows if r["rtf"] is not None)
        dry_wets = [
            float(r["dry_wet_pct"]) for r in t_rows if r["dry_wet_pct"] is not None
        ]

        p50_rtf = round(_percentile(rtfs, 50.0), 6) if rtfs else 0.0
        p99_rtf = round(_percentile(rtfs, 99.0), 6) if rtfs else 0.0
        avg_dry_wet_pct = (
            round(sum(dry_wets) / len(dry_wets), 2) if dry_wets else 0.0
        )
        thumbs_up_rate = round(f_clean / f_total, 4) if f_total > 0 else 0.0

        recent_ratings = [
            {
                "id": int(r["id"]),
                "created_at": r["created_at"],
                "model_version": r["model_version"],
                "rating": r["rating"],
                "dry_wet_pct": float(r["dry_wet_pct"]),
                "opt_in_clip_path": r["opt_in_clip_path"],
            }
            for r in recent_rows
        ]

        return {
            "total_sessions": total_sessions,
            "webgpu_count": webgpu_count,
            "wasm_count": wasm_count,
            "p50_rtf": p50_rtf,
            "p99_rtf": p99_rtf,
            "avg_dry_wet_pct": avg_dry_wet_pct,
            "thumbs_up_rate": thumbs_up_rate,
            "opt_in_clips_count": int(opt_in_count),
            "recent_ratings": recent_ratings,
        }


# Default module-level singleton store and convenience functions
_default_store: Optional[TelemetryStore] = None


def get_store() -> TelemetryStore:
    global _default_store
    if _default_store is None:
        _default_store = TelemetryStore()
    return _default_store


def save_opt_in_clip(audio_bytes: bytes) -> str:
    return get_store().save_opt_in_clip(audio_bytes)


def log_telemetry(**kwargs: Any) -> int:
    return get_store().log_telemetry(**kwargs)


def log_feedback(**kwargs: Any) -> int:
    return get_store().log_feedback(**kwargs)


def get_metrics_dashboard(recent_limit: int = 20) -> Dict[str, Any]:
    return get_store().get_metrics_dashboard(recent_limit=recent_limit)
