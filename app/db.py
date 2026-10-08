"""DuckDB access for the API (read-only; the pipeline is the only writer)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

from app.errors import DependencyUnavailableError

logger = logging.getLogger(__name__)


class Database:
    """Owns one read-only connection; hands out a cursor per unit of work.

    ``connection.cursor()`` gives each request its own DuckDB connection to the
    same database instance, which is the documented way to use DuckDB safely
    from multiple threads (FastAPI runs sync endpoints in a thread pool).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._conn: duckdb.DuckDBPyConnection | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_connected(self) -> bool:
        return self._conn is not None

    def connect(self) -> None:
        if not self._path.is_file():
            raise FileNotFoundError(f"Warehouse not found at {self._path} — run the pipeline first")
        self._conn = duckdb.connect(str(self._path), read_only=True)
        logger.info("db.connected", extra={"db_path": str(self._path)})

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def cursor(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if self._conn is None:
            raise DependencyUnavailableError("Database is not available", {"db_path": str(self._path)})
        cur = self._conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def fetch_dicts(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            rel = cur.execute(sql, params or [])
            cols = [d[0] for d in rel.description]
            return [dict(zip(cols, row, strict=True)) for row in rel.fetchall()]

    def fetch_one(self, sql: str, params: list[Any] | None = None) -> dict[str, Any] | None:
        rows = self.fetch_dicts(sql, params)
        return rows[0] if rows else None

    def ping(self) -> dict[str, Any]:
        """Readiness probe: proves the file is readable and the curated layer exists."""
        row = self.fetch_one(
            "SELECT (SELECT count(*) FROM curated.shipments) AS shipments, run_id, finished_at FROM meta.pipeline_run"
        )
        return row or {}
