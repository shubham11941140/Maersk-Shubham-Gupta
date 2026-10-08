"""Business logic for shipment lookups and listing."""

from __future__ import annotations

import math
from typing import Any

from app.errors import InvalidRequestError, ShipmentNotFoundError
from app.repositories.shipment_repository import PageRequest, ShipmentFilters, ShipmentRepository
from app.schemas.common import Pagination
from app.schemas.shipments import PortSummary, ShipmentDetail, ShipmentListResponse, ShipmentSummary


class ShipmentService:
    def __init__(self, repository: ShipmentRepository) -> None:
        self._repo = repository

    def get(self, shipment_id: str) -> ShipmentDetail:
        row = self._repo.get_by_id(shipment_id)
        if row is None:
            raise ShipmentNotFoundError(shipment_id)
        return _to_detail(row)

    def list(self, filters: ShipmentFilters, page: PageRequest) -> ShipmentListResponse:
        if filters.date_from and filters.date_to and filters.date_from > filters.date_to:
            raise InvalidRequestError(
                "date_from must be on or before date_to",
                {"date_from": filters.date_from.isoformat(), "date_to": filters.date_to.isoformat()},
            )
        rows, total = self._repo.list(filters, page)
        return ShipmentListResponse(
            items=[ShipmentSummary.model_validate(r) for r in rows],
            pagination=Pagination(
                page=page.page,
                page_size=page.page_size,
                total_items=total,
                total_pages=math.ceil(total / page.page_size) if total else 0,
            ),
        )


def _to_detail(row: dict[str, Any]) -> ShipmentDetail:
    data = dict(row)
    data["origin"] = PortSummary(
        port_code=row["origin_port"],
        port_name=data.pop("origin_port_name", None),
        country=data.pop("origin_country", None),
        region=data.pop("origin_region", None),
    )
    data["destination"] = PortSummary(
        port_code=row["destination_port"],
        port_name=data.pop("destination_port_name", None),
        country=data.pop("destination_country", None),
        region=data.pop("destination_region", None),
    )
    data["dq_flags"] = list(row.get("dq_flags") or [])
    return ShipmentDetail.model_validate(data)
