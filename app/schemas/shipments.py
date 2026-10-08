from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import Pagination, ShipmentStatus


class PortSummary(BaseModel):
    port_code: str
    port_name: str | None = None
    country: str | None = None
    region: str | None = None


class ShipmentSummary(BaseModel):
    shipment_id: str
    customer_id: str
    origin_port: str
    destination_port: str
    route_key: str
    vessel_id: str
    booking_date: datetime
    planned_departure: datetime
    actual_departure: datetime | None
    planned_arrival: datetime
    actual_arrival: datetime | None
    container_count: int | None
    cargo_type: str
    weight_tons: float | None
    status: ShipmentStatus
    actual_delay_hours: float | None = Field(description="actual_arrival − planned_arrival, hours (null if unknown)")
    on_time_flag: bool | None = Field(description="true when delay ≤ 24h")
    transit_days_planned: float
    transit_days_actual: float | None


class ShipmentDetail(ShipmentSummary):
    status_raw: str | None = Field(description="Status exactly as it appeared in the source file")
    booking_lead_days: float
    origin: PortSummary
    destination: PortSummary
    dq_flags: list[str] = Field(description="Data-quality fixes / flags applied to this row during curation")


class ShipmentListResponse(BaseModel):
    items: list[ShipmentSummary]
    pagination: Pagination
