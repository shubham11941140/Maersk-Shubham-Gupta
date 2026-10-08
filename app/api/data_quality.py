from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_dq_service
from app.schemas.common import error_responses
from app.services.data_quality_service import DataQualityReportService

router = APIRouter(prefix="/data-quality", tags=["data-quality"])


@router.get(
    "/report",
    summary="Output of the standalone DQ checks (same JSON the CLI writes)",
    responses=error_responses(422, 503),
)
def data_quality_report(
    service: Annotated[DataQualityReportService, Depends(get_dq_service)],
    severity: Annotated[Literal["critical", "warning", "info"] | None, Query()] = None,
    dataset: Annotated[Literal["ports", "shipments", "port_events"] | None, Query()] = None,
    only_failed: Annotated[bool, Query(description="Hide checks that passed")] = False,
) -> dict[str, Any]:
    return service.report(severity=severity, dataset=dataset, only_failed=only_failed)
