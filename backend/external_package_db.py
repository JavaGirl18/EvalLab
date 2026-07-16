"""
External Evaluation Package registry.

Manages the external_packages table — one row per submitted package.
Lifecycle: imported → reviewed → approved → (evaluation record created)
                                           or rejected → archived
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "evallab.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_external_package_tables() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _expkg_seq (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS external_packages (
                pkg_id           TEXT PRIMARY KEY,
                status           TEXT NOT NULL DEFAULT 'imported',
                source_label     TEXT NOT NULL DEFAULT '',
                submitted_by     TEXT NOT NULL DEFAULT '',
                received_at      TEXT NOT NULL,
                reviewed_at      TEXT,
                approved_at      TEXT,
                rejected_at      TEXT,
                rejection_reason TEXT,
                file_manifest    TEXT NOT NULL DEFAULT '[]',
                detected_meta    TEXT NOT NULL DEFAULT '{}',
                mapped_meta      TEXT NOT NULL DEFAULT '{}',
                notes            TEXT NOT NULL DEFAULT '',
                evaluation_id    TEXT,
                storage_dir      TEXT NOT NULL
            )
        """)


def _next_pkg_id() -> str:
    with _conn() as conn:
        cursor = conn.execute("INSERT INTO _expkg_seq DEFAULT VALUES")
        seq = cursor.lastrowid
    return f"EXPKG-{seq:06d}"


def create_package(
    source_label: str,
    submitted_by: str,
    file_manifest: list,
    detected_meta: dict,
    storage_dir: str,
) -> str:
    pkg_id = _next_pkg_id()
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """INSERT INTO external_packages
               (pkg_id, source_label, submitted_by, received_at,
                file_manifest, detected_meta, mapped_meta, storage_dir)
               VALUES (?,?,?,?,?,?,?,?)""",
            (pkg_id, source_label, submitted_by, now,
             json.dumps(file_manifest), json.dumps(detected_meta),
             json.dumps({}), storage_dir),
        )
    return pkg_id


def _deserialize(d: dict) -> dict:
    for key in ("file_manifest", "detected_meta", "mapped_meta"):
        if isinstance(d.get(key), str):
            try:
                d[key] = json.loads(d[key])
            except Exception:
                d[key] = {} if key != "file_manifest" else []
    return d


def get_package(pkg_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM external_packages WHERE pkg_id = ?", (pkg_id,)
        ).fetchone()
    return _deserialize(dict(row)) if row else None


def list_packages() -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM external_packages ORDER BY received_at DESC"
        ).fetchall()
    return [_deserialize(dict(r)) for r in rows]


def update_meta(pkg_id: str, mapped_meta: dict, notes: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """UPDATE external_packages
               SET mapped_meta=?, notes=?, status=CASE
                   WHEN status='imported' THEN 'reviewed'
                   ELSE status END,
               reviewed_at=CASE
                   WHEN status='imported' THEN ?
                   ELSE reviewed_at END
               WHERE pkg_id=?""",
            (json.dumps(mapped_meta), notes, now, pkg_id),
        )


def mark_approved(pkg_id: str, evaluation_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """UPDATE external_packages
               SET status='approved', approved_at=?, evaluation_id=?
               WHERE pkg_id=?""",
            (now, evaluation_id, pkg_id),
        )


def mark_rejected(pkg_id: str, reason: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """UPDATE external_packages
               SET status='rejected', rejected_at=?, rejection_reason=?
               WHERE pkg_id=?""",
            (now, reason, pkg_id),
        )
