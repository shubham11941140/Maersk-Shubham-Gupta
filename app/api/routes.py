from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query

from app.dependencies import get_route_service
from app.schemas.common import PortCode, error_responses
from app.schemas.routes import RouteStatsResponse
from app.services.route_service import RouteService

router = APIRouter(prefix="/routes", tags=["routes"])


@router.get(
    "/{origin}/{destination}/stats",
    response_model=RouteStatsResponse,
    summary="Aggregated delay / on-time metrics for a route",
    responses=error_responses(404, 422, 503),
)
def route_stats(
    origin: Annotated[PortCode, Path(description="Origin port code")],
    destination: Annotated[PortCode, Path(description="Destination port code")],
    service: Annotated[RouteService, Depends(get_route_service)],
    date_from: Annotated[date | None, Query(description="planned_departure on/after")] = None,
    date_to: Annotated[date | None, Query(description="planned_departure on/before")] = None,
) -> RouteStatsResponse:
    return service.stats(origin, destination, date_from, date_to)
