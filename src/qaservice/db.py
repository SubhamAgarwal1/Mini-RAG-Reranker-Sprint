
"""SQLite helpers for storing sources and chunks."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Dict, Any

from . import config

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    file_name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    char_len INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    FOREIGN KEY(source_id) REFERENCES sources(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_source_chunk
    ON chunks(source_id, chunk_index);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);
"""


@contextmanager
def get_connection(db_path: Path | None = None):
    """Context manager that yields a SQLite3 connection."""
    target = db_path or config.SQLITE_DB
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.commit()
        conn.close()


def initialise_database(conn: sqlite3.Connection) -> None:
    """Create core tables if they do not already exist."""
    conn.executescript(SCHEMA_SQL)


def load_sources() -> List[Dict[str, str]]:
    """Load the sources JSON file."""
    with open(config.SOURCES_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def upsert_sources(conn: sqlite3.Connection, sources: List[Dict[str, str]]) -> None:
    """Insert or replace the sources metadata."""
    for src in sources:
        conn.execute(
            """
            INSERT INTO sources (title, url, file_name)
            VALUES (:title, :url, :file_name)
            ON CONFLICT(file_name) DO UPDATE SET
                title=excluded.title,
                url=excluded.url
            """,
            src,
        )


def insert_chunks(conn: sqlite3.Connection, chunk_rows: Iterable[Dict[str, Any]]) -> None:
    """Bulk insert chunk rows and refresh the FTS index."""
    conn.execute("DELETE FROM chunks")
    conn.executemany(
        """
        INSERT INTO chunks (source_id, chunk_index, text, char_len, page_start, page_end)
        VALUES (:source_id, :chunk_index, :text, :char_len, :page_start, :page_end)
        """,
        chunk_rows,
    )
    conn.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('delete-all')")
    conn.execute(
        "INSERT INTO chunk_fts(rowid, text) SELECT id, text FROM chunks"
    )


def fetch_chunks_by_ids(conn: sqlite3.Connection, ids: List[int]) -> Dict[int, sqlite3.Row]:
    if not ids:
        return {}
    cursor = conn.execute(
        "SELECT id, source_id, chunk_index, text, char_len, page_start, page_end FROM chunks WHERE id IN (%s)" %
        ",".join("?" for _ in ids),
        ids,
    )
    return {row["id"]: row for row in cursor.fetchall()}


def fetch_source_map(conn: sqlite3.Connection) -> Dict[int, sqlite3.Row]:
    cursor = conn.execute("SELECT id, title, url, file_name FROM sources")
    return {row["id"]: row for row in cursor.fetchall()}
