from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Path, Query

from app.config import Settings
from app.dependencies import get_settings_dep, get_shipment_service
from app.errors import InvalidRequestError
from app.repositories.shipment_repository import SORTABLE_COLUMNS, PageRequest, ShipmentFilters
from app.schemas.common import PortCode, ShipmentStatus, error_responses
from app.schemas.shipments import ShipmentDetail, ShipmentListResponse
from app.services.shipment_service import ShipmentService

router = APIRouter(prefix="/shipments", tags=["shipments"])

SortColumn = Literal[SORTABLE_COLUMNS]  # type: ignore[valid-type]


@router.get(
    "",
    response_model=ShipmentListResponse,
    summary="List shipments with filters and pagination",
    responses=error_responses(422, 503),
)
def list_shipments(
    service: Annotated[ShipmentService, Depends(get_shipment_service)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
    origin: Annotated[PortCode | None, Query(description="Origin port code, e.g. CNSHA")] = None,
    destination: Annotated[PortCode | None, Query(description="Destination port code, e.g. NLRTM")] = None,
    status: Annotated[ShipmentStatus | None, Query(description="Curated status")] = None,
    cargo_type: Annotated[str | None, Query(max_length=64)] = None,
    date_from: Annotated[date | None, Query(description="planned_departure on/after (YYYY-MM-DD)")] = None,
    date_to: Annotated[date | None, Query(description="planned_departure on/before (YYYY-MM-DD)")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int | None, Query(ge=1)] = None,
    sort_by: Annotated[SortColumn, Query()] = "planned_departure",
    order: Annotated[Literal["asc", "desc"], Query()] = "asc",
) -> ShipmentListResponse:
    size = page_size or settings.default_page_size
    if size > settings.max_page_size:
        raise InvalidRequestError(
            f"page_size must be ≤ {settings.max_page_size}", {"max_page_size": settings.max_page_size}
        )
    filters = ShipmentFilters(
        origin=origin,
        destination=destination,
        status=status.value if status else None,
        cargo_type=cargo_type,
        date_from=date_from,
        date_to=date_to,
    )
    return service.list(filters, PageRequest(page=page, page_size=size, sort_by=sort_by, descending=order == "desc"))


@router.get(
    "/{shipment_id}",
    response_model=ShipmentDetail,
    summary="Full details for one shipment",
    responses=error_responses(404, 422, 503),
)
def get_shipment(
    shipment_id: Annotated[str, Path(description="e.g. SHP-00421", max_length=32)],
    service: Annotated[ShipmentService, Depends(get_shipment_service)],
) -> ShipmentDetail:
    return service.get(shipment_id.strip().upper())
