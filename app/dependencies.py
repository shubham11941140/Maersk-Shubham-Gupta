"""FastAPI dependency providers (composition root lives in ``app.main``).

Everything is resolved from ``app.state`` so tests can inject fakes with
``app.dependency_overrides`` without touching globals.
"""

from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.repositories.shipment_repository import DuckDBShipmentRepository, ShipmentRepository
from app.services.data_quality_service import DataQualityReportService
from app.services.health_service import HealthService
from app.services.prediction_service import PredictionService
from app.services.route_service import RouteService
from app.services.shipment_service import ShipmentService


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_repository(request: Request) -> ShipmentRepository:
    return DuckDBShipmentRepository(request.app.state.db)


def get_shipment_service(request: Request) -> ShipmentService:
    return ShipmentService(get_repository(request))


def get_route_service(request: Request) -> RouteService:
    return RouteService(get_repository(request))


def get_prediction_service(request: Request) -> PredictionService:
    return PredictionService(request.app.state.predictor, get_repository(request))


def get_dq_service(request: Request) -> DataQualityReportService:
    return request.app.state.dq_service


def get_health_service(request: Request) -> HealthService:
    state = request.app.state
    return HealthService(state.settings, state.db, state.predictor, state.dq_service, state.started_at)
