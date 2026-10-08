from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.dependencies import get_analytics_service
from app.schemas.analytics import PortListResponse
from app.schemas.common import error_responses
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/ports", tags=["ports"])


@router.get("", response_model=PortListResponse, summary="Port reference data", responses=error_responses(503))
def list_ports(service: Annotated[AnalyticsService, Depends(get_analytics_service)]) -> PortListResponse:
    return service.ports()
