"""SQLite storage for the cleaned dataset.

A single file keeps the data versionable (its ``meta`` table records the
source commit and cleaning report) without needing a database server.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE documents (
    doc_id TEXT NOT NULL,
    split TEXT NOT NULL,
    ticker TEXT,
    quarter INTEGER,
    year INTEGER,
    transcript TEXT NOT NULL,
    summary TEXT NOT NULL,
    n_words_transcript INTEGER NOT NULL,
    n_words_summary INTEGER NOT NULL,
    n_summary_lines INTEGER NOT NULL,
    PRIMARY KEY (split, doc_id)
);
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_COLUMNS = [
    "doc_id", "split", "ticker", "quarter", "year", "transcript", "summary",
    "n_words_transcript", "n_words_summary", "n_summary_lines",
]


def write_db(df: pd.DataFrame, path: Path, meta: dict) -> None:
    """Write ``df`` and ``meta`` to a fresh SQLite file, replacing any old one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)
        df[_COLUMNS].to_sql("documents", conn, if_exists="append", index=False)
        conn.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [(k, json.dumps(v)) for k, v in meta.items()],
        )
    conn.close()


def load_split(path: Path, split: str, limit: int | None = None) -> pd.DataFrame:
    """Load one split, ordered by ``doc_id`` so ``limit`` always picks the same rows."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_data.py first")
    query = "SELECT * FROM documents WHERE split = ? ORDER BY doc_id"
    params: list = [split]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def load_meta(path: Path) -> dict:
    with sqlite3.connect(path) as conn:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    conn.close()
    return {k: json.loads(v) for k, v in rows}
