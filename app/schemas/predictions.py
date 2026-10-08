from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self

from pydantic import BaseModel, Field, StringConstraints, model_validator

from app.schemas.common import PortCode, ShipmentId


class BookingFeatures(BaseModel):
    """Everything the model needs, all of it known at booking time."""

    origin_port: PortCode
    destination_port: PortCode
    cargo_type: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    container_count: int | None = Field(default=None, ge=1, le=1000)
    weight_tons: float | None = Field(default=None, gt=0, le=100_000)
    booking_date: datetime
    planned_departure: datetime
    planned_arrival: datetime

    @model_validator(mode="after")
    def _check_sequence(self) -> Self:
        if self.origin_port == self.destination_port:
            raise ValueError("origin_port and destination_port must differ")
        if self.planned_departure < self.booking_date:
            raise ValueError("planned_departure must not be before booking_date")
        if self.planned_arrival <= self.planned_departure:
            raise ValueError("planned_arrival must be after planned_departure")
        return self


class PredictDelayRequest(BaseModel):
    """Provide exactly one of ``shipment_id`` (look up an existing booking) or ``booking``."""

    shipment_id: ShipmentId | None = None
    booking: BookingFeatures | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"shipment_id": "SHP-00421"},
                {
                    "booking": {
                        "origin_port": "CNSHA",
                        "destination_port": "NLRTM",
                        "cargo_type": "Electronics",
                        "container_count": 20,
                        "weight_tons": 950.5,
                        "booking_date": "2025-09-01T08:00:00",
                        "planned_departure": "2025-09-10T08:00:00",
                        "planned_arrival": "2025-10-12T08:00:00",
                    }
                },
            ]
        }
    }

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.shipment_id is None) == (self.booking is None):
            raise ValueError("Provide exactly one of 'shipment_id' or 'booking'")
        return self


class PredictDelayResponse(BaseModel):
    shipment_id: str | None
    delay_probability: float = Field(ge=0, le=1, description="P(arrival > 24h late)")
    predicted_delayed: bool
    risk_band: str = Field(examples=["LOW", "MEDIUM", "HIGH"])
    decision_threshold: float
    model_version: str
    inputs: BookingFeatures
