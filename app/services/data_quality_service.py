"""Serves the report produced by the standalone DQ module (``python -m dq``)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from app.errors import DependencyUnavailableError


class DataQualityReportService:
    """Reads the JSON report from disk, caching it until the file changes (mtime)."""

    def __init__(self, report_path: Path) -> None:
        self._path = Path(report_path)
        self._lock = threading.Lock()
        self._cache: tuple[float, dict[str, Any]] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, Any]:
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError as exc:
            raise DependencyUnavailableError(
                "Data-quality report has not been generated yet", {"expected_path": str(self._path)}
            ) from exc
        with self._lock:
            if self._cache is None or self._cache[0] != mtime:
                try:
                    report = json.loads(self._path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise DependencyUnavailableError("Data-quality report is corrupt") from exc
                self._cache = (mtime, report)
            return self._cache[1]

    def report(
        self, severity: str | None = None, dataset: str | None = None, only_failed: bool = False
    ) -> dict[str, Any]:
        report = self._load()
        checks = [
            c
            for c in report.get("checks", [])
            if (severity is None or c["severity"] == severity)
            and (dataset is None or c["dataset"] == dataset)
            and (not only_failed or c["status"] != "pass")
        ]
        return {**report, "checks": checks}

    def summary(self) -> dict[str, Any]:
        report = self._load()
        return {"generated_at": report.get("generated_at"), **report.get("summary", {})}
