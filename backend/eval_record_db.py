"""
Evaluation Record registry.

Manages the evaluation_records table, which gives every completed batch run
a stable human-readable ID (EV-{YYYY}-{NNNNNN}) and optionally stores a
frozen JSON snapshot so that future exports are immune to metadata changes.
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


def init_eval_record_tables() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _ev_seq (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_records (
                evaluation_id      TEXT PRIMARY KEY,
                run_key            TEXT UNIQUE NOT NULL,
                registered_at      TEXT NOT NULL,
                first_exported_at  TEXT,
                export_count       INTEGER NOT NULL DEFAULT 0,
                record_snapshot    TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_er_run_key ON evaluation_records(run_key)"
        )


def _next_evaluation_id() -> str:
    year = datetime.now(timezone.utc).year
    with _conn() as conn:
        cursor = conn.execute("INSERT INTO _ev_seq DEFAULT VALUES")
        seq = cursor.lastrowid
    return f"EV-{year}-{seq:06d}"


def register_run_key(run_key: str) -> str:
    """
    Return an evaluation_id for run_key, creating one if it doesn't exist yet.
    Safe to call multiple times — idempotent.
    """
    with _conn() as conn:
        row = conn.execute(
            "SELECT evaluation_id FROM evaluation_records WHERE run_key = ?",
            (run_key,),
        ).fetchone()
        if row:
            return row["evaluation_id"]

    ev_id = _next_evaluation_id()
    now   = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        try:
            conn.execute(
                "INSERT INTO evaluation_records (evaluation_id, run_key, registered_at) VALUES (?, ?, ?)",
                (ev_id, run_key, now),
            )
        except sqlite3.IntegrityError:
            # Race condition — another request already inserted it
            row = conn.execute(
                "SELECT evaluation_id FROM evaluation_records WHERE run_key = ?",
                (run_key,),
            ).fetchone()
            return row["evaluation_id"]
    return ev_id


def get_by_evaluation_id(evaluation_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM evaluation_records WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()
        return dict(row) if row else None


def get_by_run_key(run_key: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM evaluation_records WHERE run_key = ?",
            (run_key,),
        ).fetchone()
        return dict(row) if row else None


def list_evaluation_records(
    experiment_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    List registered evaluation records, optionally filtered by experiment or batch.
    Joins against batch_run_items to resolve experiment_id / batch_id filters.
    """
    if experiment_id or batch_id:
        clauses, params = [], []
        if experiment_id:
            clauses.append("i.experiment_id = ?")
            params.append(experiment_id)
        if batch_id:
            clauses.append("i.batch_id = ?")
            params.append(batch_id)
        where = " AND ".join(clauses)
        with _conn() as conn:
            rows = conn.execute(
                f"""
                SELECT er.*
                FROM evaluation_records er
                JOIN batch_run_items i ON er.run_key = i.run_key
                WHERE {where}
                ORDER BY er.registered_at DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
    else:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM evaluation_records ORDER BY registered_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def freeze_snapshot(evaluation_id: str, record_json: dict) -> None:
    """Store the frozen JSON snapshot and record the first export timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        row = conn.execute(
            "SELECT first_exported_at FROM evaluation_records WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()
        first = row["first_exported_at"] if row else None
        conn.execute(
            """
            UPDATE evaluation_records
            SET record_snapshot   = ?,
                first_exported_at = ?,
                export_count      = export_count + 1
            WHERE evaluation_id   = ?
            """,
            (
                json.dumps(record_json),
                first or now,
                evaluation_id,
            ),
        )


def increment_export_count(evaluation_id: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE evaluation_records SET export_count = export_count + 1 WHERE evaluation_id = ?",
            (evaluation_id,),
        )
