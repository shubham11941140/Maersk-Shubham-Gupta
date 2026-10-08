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


# --------------------------------------------------------------------------- #
# Analytics (ports / rankings)
# --------------------------------------------------------------------------- #
PERIODS = [
    {"period": "2025-Q2", "complete": True, "shipments": 10, "completed": 9},
    {"period": "2025-Q3", "complete": True, "shipments": 12, "completed": 11},
    {"period": "2025-Q4", "complete": False, "shipments": 3, "completed": 3},
]


@dataclass
class FakeAnalyticsRepository:
    periods_rows: list[dict[str, Any]] = field(default_factory=lambda: list(PERIODS))
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def list_ports(self) -> list[dict[str, Any]]:
        return [
            {
                "port_code": "CNSHA",
                "port_name": "Shanghai",
                "country": "China",
                "region": "APAC",
                "timezone": "UTC+8",
                "avg_congestion_score": 0.85,
            }
        ]

    def periods(self) -> list[dict[str, Any]]:
        return self.periods_rows

    def route_rankings(self, metric, descending, period, min_completed, limit):
        self.calls.append(("route_rankings", (metric, descending, period, min_completed, limit)))
        return [
            {
                "route_key": "CNSHA → NLRTM",
                "origin_port": "CNSHA",
                "destination_port": "NLRTM",
                "shipment_count": 5,
                "completed_count": 4,
                "avg_delay_hours": 50.123,
                "on_time_rate": 0.25,
            }
        ]


# --------------------------------------------------------------------------- #
# Assistant
# --------------------------------------------------------------------------- #
def tool_use(name: str, args: dict[str, Any], call_id: str = "tu_1") -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": args}


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


class ScriptedLLM:
    """Replays a fixed list of responses (or callables(messages) -> blocks) and records every request."""

    def __init__(self, script: list[Any], model: str = "gpt-5.6-luna") -> None:
        from assistant.llm import LLMResponse, Usage  # local import: tests/fakes is imported by API tests too

        self._LLMResponse, self._Usage = LLMResponse, Usage
        self.model = model
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def create(self, system, messages, tools, allow_tools=True):
        import copy

        self.requests.append(
            {"system": system, "messages": copy.deepcopy(messages), "tools": tools, "allow_tools": allow_tools}
        )
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of responses")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        blocks = step(messages) if callable(step) else step
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return self._LLMResponse(content=blocks, stop_reason=stop, usage=self._Usage(1000, 100), model=self.model)


def last_tool_result(messages: list[dict[str, Any]]) -> dict[str, Any]:
    import json

    return json.loads(messages[-1]["content"][-1]["content"])


@dataclass
class FakeSupplyChainApi:
    calls: list[tuple[str, Any]] = field(default_factory=list)
    fail_with: Exception | None = None

    def _record(self, name: str, payload: Any) -> None:
        self.calls.append((name, payload))
        if self.fail_with:
            raise self.fail_with

    def ports(self):
        return FakeAnalyticsRepository().list_ports()

    def get_shipment(self, shipment_id):
        self._record("get_shipment", shipment_id)
        return make_shipment(shipment_id) | {"origin": {"port_code": "CNSHA"}, "destination": {"port_code": "NLRTM"}}

    def list_shipments(self, params):
        self._record("list_shipments", params)
        items = [make_shipment(f"SHP-0000{i}") for i in range(1, 4)]
        return {"items": items, "pagination": {"total_items": 42}}

    def route_stats(self, origin, destination, params):
        self._record("route_stats", (origin, destination, params))
        return {"route_key": f"{origin} → {destination}", "on_time_rate": 0.75, "completed_count": 12}

    def route_rankings(self, params):
        self._record("route_rankings", params)
        return {"period": "2025-Q3", "available_periods": PERIODS, "items": []}

    def predict_delay(self, shipment_id):
        self._record("predict_delay", shipment_id)
        return {
            "shipment_id": shipment_id,
            "delay_probability": 0.42,
            "predicted_delayed": False,
            "risk_band": "MEDIUM",
            "decision_threshold": 0.5,
            "model_version": "test",
            "inputs": {},
        }
