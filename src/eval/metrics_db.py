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
  ndcg_at_5 REAL,
  faithfulness REAL,
  notes TEXT,
  k INTEGER,
  recall_at_k REAL,
  precision_at_k REAL,
  ndcg_at_k REAL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(eval_runs)")}
    if "ndcg_at_5" not in existing_cols:
        conn.execute("ALTER TABLE eval_runs ADD COLUMN ndcg_at_5 REAL")
    for col in ("k", "recall_at_k", "precision_at_k", "ndcg_at_k"):
        if col not in existing_cols:
            col_type = "INTEGER" if col == "k" else "REAL"
            conn.execute(f"ALTER TABLE eval_runs ADD COLUMN {col} {col_type}")
    conn.commit()
    return conn


def record_run(run_id: str, timestamp: str, config_hash: str, recall_at_5, precision_at_5,
                mrr, ndcg_at_5, faithfulness, notes: str,
                k: int | None = None, recall_at_k: float | None = None,
                precision_at_k: float | None = None, ndcg_at_k: float | None = None):
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs
                (run_id, timestamp, config_hash, recall_at_5, precision_at_5, mrr, ndcg_at_5,
                 faithfulness, notes, k, recall_at_k, precision_at_k, ndcg_at_k)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, timestamp, config_hash, recall_at_5, precision_at_5, mrr, ndcg_at_5,
             faithfulness, notes, k, recall_at_k, precision_at_k, ndcg_at_k),
        )
    conn.close()


def all_runs():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM eval_runs ORDER BY timestamp").fetchall()
    conn.close()
    return rows
