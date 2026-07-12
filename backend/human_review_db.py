import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "evallab.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_human_review_tables() -> None:
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS _hr_seq (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS human_reviews (
                review_id           TEXT PRIMARY KEY,
                experiment_id       TEXT NOT NULL,
                batch_id            TEXT NOT NULL,
                run_key             TEXT NOT NULL,
                dataset_item_id     TEXT NOT NULL,
                prompt_variant      TEXT NOT NULL,
                transformation_task TEXT NOT NULL,
                subject_model       TEXT NOT NULL,
                judge_model         TEXT NOT NULL,
                taxonomy_version    TEXT NOT NULL DEFAULT 'v1',
                rubric_version      TEXT NOT NULL DEFAULT 'v1',
                review_round        INTEGER NOT NULL DEFAULT 1,
                selection_reasons   TEXT NOT NULL DEFAULT '[]',
                review_status       TEXT NOT NULL DEFAULT 'pending',
                blinded             INTEGER NOT NULL DEFAULT 1,
                packet_snapshot     TEXT NOT NULL,
                created_at          TEXT NOT NULL,
                exported_at         TEXT,
                completed_at        TEXT,
                UNIQUE(run_key, review_round)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS human_review_responses (
                response_id                TEXT PRIMARY KEY,
                review_id                  TEXT NOT NULL REFERENCES human_reviews(review_id),
                reviewer_id                TEXT NOT NULL,
                agreement_with_judge       TEXT NOT NULL,
                disagreement_types         TEXT NOT NULL DEFAULT '[]',
                comments                   TEXT DEFAULT '',
                missed_failures            TEXT DEFAULT '',
                incorrectly_flagged        TEXT DEFAULT '',
                preserved_meaning          TEXT DEFAULT '',
                cultural_context_preserved TEXT DEFAULT '',
                additional_comments        TEXT DEFAULT '',
                reviewed_at                TEXT NOT NULL,
                imported_at                TEXT NOT NULL,
                import_source              TEXT NOT NULL DEFAULT 'csv',
                raw_import                 TEXT NOT NULL DEFAULT '{}',
                UNIQUE(review_id, reviewer_id)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hr_batch ON human_reviews(batch_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hr_experiment ON human_reviews(experiment_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hrr_review ON human_review_responses(review_id)"
        )
        conn.commit()


def _next_review_id() -> str:
    with _conn() as conn:
        cursor = conn.execute("INSERT INTO _hr_seq DEFAULT VALUES")
        seq = cursor.lastrowid
        conn.commit()
    return f"HR-{seq:06d}"


def create_human_review(
    *,
    experiment_id: str,
    batch_id: str,
    run_key: str,
    dataset_item_id: str,
    prompt_variant: str,
    transformation_task: str,
    subject_model: str,
    judge_model: str,
    taxonomy_version: str,
    rubric_version: str,
    review_round: int,
    selection_reasons: list,
    blinded: bool,
    packet_snapshot: dict,
) -> Optional[str]:
    """Returns review_id if created, None if (run_key, review_round) already exists."""
    review_id = _next_review_id()
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _conn() as conn:
            conn.execute(
                """INSERT INTO human_reviews
                   (review_id, experiment_id, batch_id, run_key, dataset_item_id,
                    prompt_variant, transformation_task, subject_model, judge_model,
                    taxonomy_version, rubric_version, review_round, selection_reasons,
                    review_status, blinded, packet_snapshot, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    review_id, experiment_id, batch_id, run_key, dataset_item_id,
                    prompt_variant, transformation_task, subject_model, judge_model,
                    taxonomy_version, rubric_version, review_round,
                    json.dumps(selection_reasons),
                    "pending", int(blinded),
                    json.dumps(packet_snapshot, ensure_ascii=False), now,
                ),
            )
            conn.commit()
        return review_id
    except sqlite3.IntegrityError:
        return None


def get_human_review(review_id: str) -> Optional[dict]:
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM human_reviews WHERE review_id = ?", (review_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["selection_reasons"] = json.loads(d.get("selection_reasons") or "[]")
        d["packet_snapshot"]   = json.loads(d.get("packet_snapshot") or "{}")
        d["blinded"] = bool(d["blinded"])
        return d


def list_human_reviews(
    batch_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    status: Optional[str] = None,
    review_round: Optional[int] = None,
) -> list:
    clauses, params = [], []
    if batch_id:
        clauses.append("batch_id = ?"); params.append(batch_id)
    if experiment_id:
        clauses.append("experiment_id = ?"); params.append(experiment_id)
    if status:
        clauses.append("review_status = ?"); params.append(status)
    if review_round is not None:
        clauses.append("review_round = ?"); params.append(review_round)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT review_id, experiment_id, batch_id, run_key, dataset_item_id, "
            f"prompt_variant, transformation_task, subject_model, judge_model, "
            f"taxonomy_version, rubric_version, review_round, selection_reasons, "
            f"review_status, blinded, created_at, exported_at, completed_at, "
            f"json_extract(packet_snapshot,'$.source_title') as source_title, "
            f"json_extract(packet_snapshot,'$.benchmark_category') as benchmark_category, "
            f"json_extract(packet_snapshot,'$.cultural_significance') as cultural_significance "
            f"FROM human_reviews {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["selection_reasons"] = json.loads(d.get("selection_reasons") or "[]")
        d["blinded"] = bool(d["blinded"])
        results.append(d)
    return results


def count_reviews(batch_id: str) -> dict:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT review_status, COUNT(*) as n FROM human_reviews WHERE batch_id=? GROUP BY review_status",
            (batch_id,),
        ).fetchall()
    counts = {r["review_status"]: r["n"] for r in rows}
    return {
        "pending":   counts.get("pending", 0),
        "exported":  counts.get("exported", 0),
        "completed": counts.get("completed", 0),
        "archived":  counts.get("archived", 0),
        "total":     sum(counts.values()),
    }


def mark_review_exported(review_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "UPDATE human_reviews SET review_status='exported', exported_at=? "
            "WHERE review_id=? AND review_status='pending'",
            (now, review_id),
        )
        conn.commit()


def mark_review_completed(review_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "UPDATE human_reviews SET review_status='completed', completed_at=? WHERE review_id=?",
            (now, review_id),
        )
        conn.commit()


def mark_review_archived(review_id: str) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE human_reviews SET review_status='archived' WHERE review_id=?",
            (review_id,),
        )
        conn.commit()


def run_key_has_review(run_key: str, review_round: int) -> bool:
    with _conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM human_reviews WHERE run_key=? AND review_round=?",
            (run_key, review_round),
        ).fetchone()
    return row is not None


def upsert_review_response(
    *,
    review_id: str,
    reviewer_id: str,
    agreement_with_judge: str,
    disagreement_types: list,
    comments: str,
    missed_failures: str,
    incorrectly_flagged: str,
    preserved_meaning: str,
    cultural_context_preserved: str,
    additional_comments: str,
    reviewed_at: str,
    imported_at: str,
    import_source: str,
    raw_import: dict,
    overwrite: bool = False,
) -> tuple:
    """Returns (response_id, was_updated). Raises ValueError on duplicate if overwrite=False."""
    with _conn() as conn:
        existing = conn.execute(
            "SELECT response_id FROM human_review_responses WHERE review_id=? AND reviewer_id=?",
            (review_id, reviewer_id),
        ).fetchone()

        if existing and not overwrite:
            raise ValueError(
                f"Response from '{reviewer_id}' for {review_id} already exists. "
                "Re-import with overwrite=true to replace."
            )

        response_id = existing["response_id"] if existing else str(uuid.uuid4())

        if existing:
            conn.execute(
                """UPDATE human_review_responses SET
                   agreement_with_judge=?, disagreement_types=?, comments=?,
                   missed_failures=?, incorrectly_flagged=?, preserved_meaning=?,
                   cultural_context_preserved=?, additional_comments=?,
                   reviewed_at=?, imported_at=?, import_source=?, raw_import=?
                   WHERE response_id=?""",
                (
                    agreement_with_judge, json.dumps(disagreement_types), comments,
                    missed_failures, incorrectly_flagged, preserved_meaning,
                    cultural_context_preserved, additional_comments,
                    reviewed_at, imported_at, import_source, json.dumps(raw_import),
                    response_id,
                ),
            )
        else:
            conn.execute(
                """INSERT INTO human_review_responses
                   (response_id, review_id, reviewer_id, agreement_with_judge,
                    disagreement_types, comments, missed_failures, incorrectly_flagged,
                    preserved_meaning, cultural_context_preserved, additional_comments,
                    reviewed_at, imported_at, import_source, raw_import)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    response_id, review_id, reviewer_id, agreement_with_judge,
                    json.dumps(disagreement_types), comments, missed_failures,
                    incorrectly_flagged, preserved_meaning, cultural_context_preserved,
                    additional_comments, reviewed_at, imported_at, import_source,
                    json.dumps(raw_import),
                ),
            )
        conn.commit()
    return response_id, bool(existing)


def get_responses_for_review(review_id: str) -> list:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM human_review_responses WHERE review_id=? ORDER BY imported_at",
            (review_id,),
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["disagreement_types"] = json.loads(d.get("disagreement_types") or "[]")
        d["raw_import"]         = json.loads(d.get("raw_import") or "{}")
        results.append(d)
    return results


def get_completed_reviews_with_responses(
    batch_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    review_round: Optional[int] = None,
) -> list:
    """All completed reviews with packet_snapshot and embedded responses, for stats."""
    clauses, params = ["review_status = 'completed'"], []
    if batch_id:
        clauses.append("batch_id = ?"); params.append(batch_id)
    if experiment_id:
        clauses.append("experiment_id = ?"); params.append(experiment_id)
    if review_round is not None:
        clauses.append("review_round = ?"); params.append(review_round)

    where = f"WHERE {' AND '.join(clauses)}"
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT review_id, experiment_id, batch_id, run_key, dataset_item_id, "
            f"prompt_variant, subject_model, review_round, selection_reasons, "
            f"review_status, packet_snapshot, created_at, exported_at, completed_at "
            f"FROM human_reviews {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["selection_reasons"] = json.loads(d.get("selection_reasons") or "[]")
        d["packet_snapshot"]   = json.loads(d.get("packet_snapshot") or "{}")
        d["responses"]         = get_responses_for_review(d["review_id"])
        results.append(d)
    return results
