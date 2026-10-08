from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class RouteStatsResponse(BaseModel):
    origin_port: str
    destination_port: str
    route_key: str
    date_from: date | None = None
    date_to: date | None = None
    shipment_count: int = Field(description="All curated shipments on the route in the period")
    completed_count: int = Field(description="Shipments with an observed arrival (basis for delay metrics)")
    cancelled_count: int
    unknown_outcome_count: int
    avg_delay_hours: float | None
    median_delay_hours: float | None
    p90_delay_hours: float | None
    max_delay_hours: float | None
    on_time_rate: float | None = Field(description="on-time / completed; null when nothing completed")
    on_time_threshold_hours: float
