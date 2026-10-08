from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from app import __version__
from app.dependencies import get_health_service
from app.schemas.health import HealthResponse
from app.services.health_service import HealthService

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Readiness check (DB, model, DQ report)",
    responses={503: {"model": HealthResponse, "description": "One or more dependencies unavailable"}},
)
def health(response: Response, service: HealthService = Depends(get_health_service)) -> HealthResponse:
    result = service.readiness()
    if result.status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get("/health/live", summary="Liveness check — the process is up")
def live() -> dict[str, str]:
    return {"status": "alive", "version": __version__}
