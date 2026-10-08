from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query

from app.dependencies import get_analytics_service, get_route_service
from app.schemas.analytics import RankingMetric, RouteRankingResponse
from app.schemas.common import PortCode, error_responses
from app.schemas.routes import RouteStatsResponse
from app.services.analytics_service import AnalyticsService
from app.services.route_service import RouteService

router = APIRouter(prefix="/routes", tags=["routes"])


@router.get(
    "/rankings",
    response_model=RouteRankingResponse,
    summary="Rank routes by delay / on-time / volume for a quarter (default: latest complete quarter)",
    responses=error_responses(422, 503),
)
def route_rankings(
    service: Annotated[AnalyticsService, Depends(get_analytics_service)],
    metric: Annotated[RankingMetric, Query()] = "avg_delay_hours",
    order: Annotated[Literal["asc", "desc"], Query()] = "desc",
    period: Annotated[
        str, Query(pattern=r"^(latest|all|\d{4}-Q[1-4])$", description="'latest', 'all' or e.g. 2025-Q3")
    ] = "latest",
    min_completed: Annotated[int, Query(ge=1, le=100, description="Exclude routes with fewer outcomes")] = 3,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> RouteRankingResponse:
    return service.route_rankings(metric, order, period, min_completed, limit)


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
