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


def init_batch_tables() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                batch_id      TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'pending',
                config        TEXT NOT NULL,
                warnings      TEXT NOT NULL DEFAULT '[]',
                created_at    TEXT NOT NULL,
                started_at    TEXT,
                completed_at  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS batch_run_items (
                run_key        TEXT PRIMARY KEY,
                batch_id       TEXT NOT NULL,
                experiment_id  TEXT NOT NULL,
                item_id        TEXT NOT NULL,
                prompt_variant TEXT NOT NULL,
                model          TEXT NOT NULL,
                status         TEXT NOT NULL DEFAULT 'pending',
                attempt        INTEGER NOT NULL DEFAULT 0,
                result_json    TEXT,
                error          TEXT,
                started_at     TEXT,
                completed_at   TEXT,
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_items_batch ON batch_run_items(batch_id)"
        )
        conn.commit()


def make_run_key(experiment_id: str, item_id: str, variant: str, model: str) -> str:
    return f"{experiment_id}|{item_id}|{variant}|{model}"


def create_batch(
    batch_id: str,
    experiment_id: str,
    config: dict,
    warnings: list[str],
    run_items: list[dict],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO batches (batch_id, experiment_id, status, config, warnings, created_at) "
            "VALUES (?, ?, 'pending', ?, ?, ?)",
            (batch_id, experiment_id, json.dumps(config), json.dumps(warnings), now),
        )
        for ri in run_items:
            rk = make_run_key(experiment_id, ri["item_id"], ri["prompt_variant"], ri["model"])
            conn.execute(
                "INSERT OR IGNORE INTO batch_run_items "
                "(run_key, batch_id, experiment_id, item_id, prompt_variant, model) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (rk, batch_id, experiment_id, ri["item_id"], ri["prompt_variant"], ri["model"]),
            )
        conn.commit()


def get_batch(batch_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        counts = conn.execute(
            "SELECT status, COUNT(*) as n FROM batch_run_items WHERE batch_id = ? GROUP BY status",
            (batch_id,),
        ).fetchall()
        count_map = {r["status"]: r["n"] for r in counts}
        d["total"]     = sum(count_map.values())
        d["completed"] = count_map.get("completed", 0)
        d["failed"]    = count_map.get("failed", 0)
        d["skipped"]   = count_map.get("skipped", 0)
        d["pending"]   = count_map.get("pending", 0) + count_map.get("running", 0)
        d["warnings"]  = json.loads(d.get("warnings") or "[]")
        return d


def get_batch_items(batch_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM batch_run_items WHERE batch_id = ? ORDER BY item_id, prompt_variant, model",
            (batch_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_batches() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("""
            SELECT
                b.batch_id, b.experiment_id, b.status,
                b.created_at, b.started_at, b.completed_at,
                COUNT(r.run_key) as total,
                SUM(CASE WHEN r.status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN r.status = 'failed'    THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN r.status IN ('pending','running') THEN 1 ELSE 0 END) as pending
            FROM batches b
            LEFT JOIN batch_run_items r ON b.batch_id = r.batch_id
            GROUP BY b.batch_id
            ORDER BY b.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]


def mark_item_running(run_key: str, attempt: int) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE batch_run_items SET status='running', attempt=?, started_at=?, error=NULL "
            "WHERE run_key=?",
            (attempt, datetime.now(timezone.utc).isoformat(), run_key),
        )
        conn.commit()


def mark_item_completed(run_key: str, result_json: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE batch_run_items SET status='completed', result_json=?, completed_at=? "
            "WHERE run_key=?",
            (result_json, datetime.now(timezone.utc).isoformat(), run_key),
        )
        conn.commit()


def mark_item_failed(run_key: str, error: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE batch_run_items SET status='failed', error=?, completed_at=? WHERE run_key=?",
            (error, datetime.now(timezone.utc).isoformat(), run_key),
        )
        conn.commit()


def mark_item_skipped(run_key: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE batch_run_items SET status='skipped', completed_at=? WHERE run_key=?",
            (datetime.now(timezone.utc).isoformat(), run_key),
        )
        conn.commit()


def mark_batch_status(
    batch_id: str,
    status: str,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> None:
    with _conn() as conn:
        updates = ["status=?"]
        params: list = [status]
        if started_at:
            updates.append("started_at=?")
            params.append(started_at)
        if completed_at:
            updates.append("completed_at=?")
            params.append(completed_at)
        params.append(batch_id)
        conn.execute(f"UPDATE batches SET {', '.join(updates)} WHERE batch_id=?", params)
        conn.commit()


def reset_stale_running_items(batch_id: str) -> None:
    """Any item stuck in 'running' from a prior crash is reset to 'pending'."""
    with _conn() as conn:
        conn.execute(
            "UPDATE batch_run_items SET status='pending' WHERE batch_id=? AND status='running'",
            (batch_id,),
        )
        conn.commit()


def cancel_batch(batch_id: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE batch_run_items SET status='cancelled' "
            "WHERE batch_id=? AND status IN ('pending','running')",
            (batch_id,),
        )
        conn.execute(
            "UPDATE batches SET status='cancelled', completed_at=? WHERE batch_id=?",
            (datetime.now(timezone.utc).isoformat(), batch_id),
        )
        conn.commit()


def get_completed_results(batch_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT result_json FROM batch_run_items WHERE batch_id=? AND status='completed'",
            (batch_id,),
        ).fetchall()
        return [json.loads(r["result_json"]) for r in rows if r["result_json"]]


def get_batch_item_by_run_key(run_key: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM batch_run_items WHERE run_key = ?", (run_key,)
        ).fetchone()
        return dict(row) if row else None
