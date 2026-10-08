"""Application factory and composition root."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import data_quality, health, ports, predictions, routes, shipments
from app.config import Settings, get_settings
from app.db import Database
from app.errors import register_exception_handlers
from app.logging_config import configure_logging
from app.middleware import RequestContextMiddleware
from app.services.data_quality_service import DataQualityReportService
from ml.predictor import ModelLoadError, SklearnDelayPredictor

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.service_name)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Start in a degraded state rather than crash-looping: /health reports
        # exactly which dependency is missing, which is far easier to operate.
        db = Database(settings.db_path)
        try:
            db.connect()
        except Exception:
            logger.exception("startup.db_unavailable", extra={"db_path": str(settings.db_path)})

        predictor = None
        try:
            predictor = SklearnDelayPredictor.from_dir(settings.model_dir)
            logger.info("startup.model_loaded", extra={"model_version": predictor.model_version})
        except ModelLoadError:
            logger.exception("startup.model_unavailable", extra={"model_dir": str(settings.model_dir)})

        app.state.settings = settings
        app.state.db = db
        app.state.predictor = predictor
        app.state.dq_service = DataQualityReportService(settings.dq_report_path)
        app.state.started_at = time.monotonic()
        logger.info("startup.complete", extra={"version": __version__, "environment": settings.environment})
        yield
        db.close()
        logger.info("shutdown.complete")

    app = FastAPI(
        title="Supply Chain Intelligence API",
        version=__version__,
        description="Shipments, route metrics, delay-risk predictions and data-quality reporting.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    for module in (health, shipments, routes, ports, predictions, data_quality):
        app.include_router(module.router)
    return app


app = create_app()
