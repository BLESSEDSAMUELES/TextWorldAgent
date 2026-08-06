"""
SQLite connection manager.

Provides a context-managed database connection with:
- WAL mode for concurrent reads
- Foreign key enforcement
- Automatic schema initialization
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


_SCHEMA_FILE = Path(__file__).parent / "schema.sql"


class DatabaseConnection:
    """Manages SQLite connection lifecycle and schema initialization."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._connection: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Create the database and apply the schema if needed."""
        conn = self._get_connection()
        schema_sql = _SCHEMA_FILE.read_text(encoding="utf-8")
        conn.executescript(schema_sql)
        conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the SQLite connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
        return self._connection

    @contextmanager
    def get_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Yield a cursor with automatic commit/rollback."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Close the database connection, checkpointing WAL first."""
        if self._connection is not None:
            try:
                # Checkpoint and switch back to DELETE journal to release WAL lock
                self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._connection.execute("PRAGMA journal_mode=DELETE")
                self._connection.commit()
            except Exception:
                pass
            self._connection.close()
            self._connection = None

    def reset(self) -> None:
        """Drop and recreate the database. Handles :memory: and Windows file-lock edge cases."""
        self.close()
        # :memory: databases have no files to remove — skip straight to (re)initialise
        if str(self._db_path) != ":memory:":
            for suffix in ("", "-wal", "-shm"):
                target = self._db_path.parent / (self._db_path.name + suffix)
                try:
                    target.unlink(missing_ok=True)
                except PermissionError:
                    # File still locked (rare on Windows); skip — initialize() will overwrite
                    pass
        self.initialize()
