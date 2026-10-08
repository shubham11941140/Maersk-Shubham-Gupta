"""Liveness / readiness logic."""

from __future__ import annotations

import time

from app import __version__
from app.config import Settings
from app.db import Database
from app.schemas.health import ComponentHealth, HealthResponse
from app.services.data_quality_service import DataQualityReportService
from ml.predictor import Predictor


class HealthService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        predictor: Predictor | None,
        dq_service: DataQualityReportService,
        started_at: float,
    ) -> None:
        self._settings = settings
        self._db = db
        self._predictor = predictor
        self._dq = dq_service
        self._started_at = started_at

    def _check_db(self) -> ComponentHealth:
        try:
            info = self._db.ping()
            return ComponentHealth(status="ok", detail={k: str(v) for k, v in info.items()})
        except Exception as exc:  # health checks must never raise
            return ComponentHealth(status="unavailable", detail={"error": str(exc)})

    def _check_model(self) -> ComponentHealth:
        if self._predictor is None:
            return ComponentHealth(status="unavailable", detail={"error": "model not loaded"})
        return ComponentHealth(status="ok", detail={"model_version": self._predictor.model_version})

    def _check_dq(self) -> ComponentHealth:
        try:
            return ComponentHealth(status="ok", detail=self._dq.summary())
        except Exception as exc:
            return ComponentHealth(status="unavailable", detail={"error": str(exc)})

    def readiness(self) -> HealthResponse:
        checks = {
            "database": self._check_db(),
            "model": self._check_model(),
            "data_quality_report": self._check_dq(),
        }
        healthy = all(c.status == "ok" for c in checks.values())
        return HealthResponse(
            status="ok" if healthy else "degraded",
            service=self._settings.service_name,
            version=__version__,
            build_sha=self._settings.build_sha,
            environment=self._settings.environment,
            uptime_seconds=round(time.monotonic() - self._started_at, 1),
            checks=checks,
        )
