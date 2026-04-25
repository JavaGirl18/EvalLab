import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "evallab.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                task TEXT NOT NULL,
                variant TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_preference(model: str, task: str, variant: str, user_prompt: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO preferences (model, task, variant, user_prompt) VALUES (?, ?, ?, ?)",
            (model, task, variant, user_prompt),
        )
        conn.commit()


def get_summary() -> dict:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT model, COUNT(*) as thumbs_up FROM preferences GROUP BY model"
        ).fetchall()
        return {row["model"]: row["thumbs_up"] for row in rows}
