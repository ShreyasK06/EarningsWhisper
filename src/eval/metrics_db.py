"""SQLite store for retrieval eval runs. One row per config change."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "metrics.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
  run_id TEXT PRIMARY KEY,
  timestamp TEXT,
  config_hash TEXT,
  recall_at_5 REAL,
  precision_at_5 REAL,
  mrr REAL,
  faithfulness REAL,
  notes TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def record_run(run_id: str, timestamp: str, config_hash: str, recall_at_5: float,
                precision_at_5: float, mrr: float, faithfulness: float | None, notes: str):
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs
                (run_id, timestamp, config_hash, recall_at_5, precision_at_5, mrr, faithfulness, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, timestamp, config_hash, recall_at_5, precision_at_5, mrr, faithfulness, notes),
        )
    conn.close()


def all_runs():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM eval_runs ORDER BY timestamp").fetchall()
    conn.close()
    return rows
