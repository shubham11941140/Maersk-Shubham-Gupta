"""In-memory test doubles for the repository and predictor Protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from app.repositories.shipment_repository import PageRequest, ShipmentFilters
from ml.predictor import Prediction


def make_shipment(shipment_id: str = "SHP-00001", **overrides: Any) -> dict[str, Any]:
    row = {
        "shipment_id": shipment_id,
        "customer_id": "CUST-1001",
        "origin_port": "CNSHA",
        "destination_port": "NLRTM",
        "route_key": "CNSHA → NLRTM",
        "vessel_id": "VSL-0001",
        "booking_date": datetime(2025, 1, 1, 8),
        "planned_departure": datetime(2025, 1, 8, 8),
        "actual_departure": datetime(2025, 1, 8, 10),
        "planned_arrival": datetime(2025, 2, 8, 8),
        "actual_arrival": datetime(2025, 2, 9, 20),
        "container_count": 12,
        "cargo_type": "Electronics",
        "weight_tons": 950.0,
        "status": "DELAYED",
        "status_raw": "DELAYED",
        "actual_delay_hours": 36.0,
        "on_time_flag": False,
        "transit_days_planned": 31.0,
        "transit_days_actual": 32.4,
        "booking_lead_days": 7.0,
        "dq_flags": [],
        "origin_port_name": "Shanghai",
        "origin_country": "China",
        "origin_region": "APAC",
        "destination_port_name": "Rotterdam",
        "destination_country": "Netherlands",
        "destination_region": "EUR",
    }
    row.update(overrides)
    return row


@dataclass
class FakeShipmentRepository:
    shipments: dict[str, dict[str, Any]] = field(default_factory=dict)
    ports: set[str] = field(default_factory=lambda: {"CNSHA", "NLRTM", "SGSIN"})
    stats: dict[str, Any] = field(default_factory=dict)
    total: int | None = None
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def get_by_id(self, shipment_id: str) -> dict[str, Any] | None:
        self.calls.append(("get_by_id", shipment_id))
        return self.shipments.get(shipment_id)

    def list(self, filters: ShipmentFilters, page: PageRequest) -> tuple[list[dict[str, Any]], int]:
        self.calls.append(("list", (filters, page)))
        rows = list(self.shipments.values())
        return rows[page.offset : page.offset + page.page_size], self.total if self.total is not None else len(rows)

    def route_stats(self, origin: str, destination: str, date_from: date | None, date_to: date | None):
        self.calls.append(("route_stats", (origin, destination, date_from, date_to)))
        return self.stats

    def existing_ports(self, codes: list[str]) -> set[str]:
        return {c for c in codes if c in self.ports}

    def get_booking_inputs(self, shipment_id: str) -> dict[str, Any] | None:
        row = self.shipments.get(shipment_id)
        if row is None:
            return None
        keys = (
            "origin_port",
            "destination_port",
            "cargo_type",
            "container_count",
            "weight_tons",
            "booking_date",
            "planned_departure",
            "planned_arrival",
        )
        return {k: row[k] for k in keys}


@dataclass
class FakePredictor:
    probability: float = 0.72
    threshold: float = 0.5
    received: list[dict[str, Any]] = field(default_factory=list)

    @property
    def model_version(self) -> str:
        return "test-model"

    def predict(self, booking: dict[str, Any]) -> Prediction:
        self.received.append(booking)
        return Prediction(
            delay_probability=self.probability,
            predicted_delayed=self.probability >= self.threshold,
            risk_band="HIGH" if self.probability >= self.threshold else "LOW",
            threshold=self.threshold,
            model_version=self.model_version,
        )
