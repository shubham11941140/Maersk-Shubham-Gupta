"""Tools exposed to the LLM. Each tool = pydantic input schema + handler over the API client.

Guardrails enforced here (deterministically, not by asking the model nicely):
* inputs are validated against the schema before any request is made;
* every tool is read-only and goes through the public, validated API;
* results are compacted and size-capped so a large answer can't flood the context;
* each successful call gets a ``source_id`` (S1, S2, …) that the answer must cite.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from assistant.api_client import ApiError, SupplyChainApi


def _port_code() -> Any:
    return Field(pattern=r"^[A-Za-z]{5}$", description="5-letter port code, e.g. CNSHA")


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace ``$ref`` pointers with their definitions so the tool schema is self-contained."""
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


_SHIPMENT_FIELDS = (
    "shipment_id", "route_key", "status", "cargo_type", "container_count", "booking_date",
    "planned_departure", "planned_arrival", "actual_arrival", "actual_delay_hours", "on_time_flag",
)  # fmt: skip


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")  # unknown arguments are an error, not silently ignored


class ShipmentFilters(_Input):
    shipment_id: str | None = Field(None, pattern=r"^SHP-\d{5}$", description="Exact shipment id, e.g. SHP-00421")
    origin: str | None = Field(None, pattern=r"^[A-Za-z]{5}$")
    destination: str | None = Field(None, pattern=r"^[A-Za-z]{5}$")
    status: Literal["DELIVERED", "DELAYED", "CANCELLED", "UNKNOWN"] | None = None
    cargo_type: str | None = Field(None, max_length=64)
    date_from: date | None = Field(None, description="planned_departure on/after (YYYY-MM-DD)")
    date_to: date | None = Field(None, description="planned_departure on/before (YYYY-MM-DD)")


class QueryShipmentsInput(_Input):
    filters: ShipmentFilters = Field(default_factory=ShipmentFilters)
    sort_by: Literal["planned_departure", "booking_date", "actual_delay_hours", "shipment_id"] = "planned_departure"
    order: Literal["asc", "desc"] = "desc"
    limit: int = Field(10, ge=1, le=25, description="Max rows returned (the total count is always returned)")


class RouteStatsInput(_Input):
    origin: str = _port_code()
    destination: str = _port_code()
    date_from: date | None = None
    date_to: date | None = None


class RankRoutesInput(_Input):
    metric: Literal["avg_delay_hours", "on_time_rate", "completed_count", "shipment_count"] = "avg_delay_hours"
    order: Literal["asc", "desc"] = "desc"
    period: str = Field(
        "latest",
        pattern=r"^(latest|all|\d{4}-Q[1-4])$",
        description="'latest' = most recent COMPLETE quarter, 'all', or e.g. '2025-Q3'",
    )
    min_completed: int = Field(3, ge=1, le=100, description="Ignore routes with fewer completed shipments")
    limit: int = Field(5, ge=1, le=20)


class PredictDelayInput(_Input):
    shipment_id: str = Field(pattern=r"^SHP-\d{5}$")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[_Input]
    handler: Callable[[SupplyChainApi, Any], dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _inline_refs(self.input_model.model_json_schema()),
        }


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def _compact(shipment: dict[str, Any]) -> dict[str, Any]:
    return {k: shipment.get(k) for k in _SHIPMENT_FIELDS}


def _query_shipments(api: SupplyChainApi, args: QueryShipmentsInput) -> dict[str, Any]:
    f = args.filters
    if f.shipment_id:
        s = api.get_shipment(f.shipment_id.upper())
        return {
            "total_matching": 1,
            "returned": 1,
            "items": [
                _compact(s)
                | {"origin": s["origin"], "destination": s["destination"], "dq_flags": s.get("dq_flags", [])}
            ],
        }
    page = api.list_shipments(
        {
            "origin": f.origin.upper() if f.origin else None,
            "destination": f.destination.upper() if f.destination else None,
            "status": f.status,
            "cargo_type": f.cargo_type,
            "date_from": f.date_from.isoformat() if f.date_from else None,
            "date_to": f.date_to.isoformat() if f.date_to else None,
            "sort_by": args.sort_by,
            "order": args.order,
            "page_size": args.limit,
        }
    )
    return {
        "total_matching": page["pagination"]["total_items"],
        "returned": len(page["items"]),
        "items": [_compact(s) for s in page["items"]],
    }


def _route_stats(api: SupplyChainApi, args: RouteStatsInput) -> dict[str, Any]:
    return api.route_stats(
        args.origin.upper(),
        args.destination.upper(),
        {
            "date_from": args.date_from.isoformat() if args.date_from else None,
            "date_to": args.date_to.isoformat() if args.date_to else None,
        },
    )


def _rank_routes(api: SupplyChainApi, args: RankRoutesInput) -> dict[str, Any]:
    r = api.route_rankings(args.model_dump())
    # available_periods is useful context but long; keep it short for the model
    r["available_periods"] = [f"{p['period']}{'' if p['complete'] else ' (partial)'}" for p in r["available_periods"]]
    return r


def _predict_delay(api: SupplyChainApi, args: PredictDelayInput) -> dict[str, Any]:
    p = api.predict_delay(args.shipment_id.upper())
    return {
        k: p[k]
        for k in (
            "shipment_id",
            "delay_probability",
            "predicted_delayed",
            "risk_band",
            "decision_threshold",
            "model_version",
        )
    } | {
        "model_quality_caveat": "The delay model has no demonstrated predictive skill on this data "
        "(test ROC-AUC ≈ 0.49, CI includes 0.5 — see docs/MODEL_CARD.md). Present the result as low-confidence.",
        "inputs": p["inputs"],
    }


TOOLS: tuple[Tool, ...] = (
    Tool(
        "query_shipments",
        "Look up shipments in the curated data product. Use filters.shipment_id for one shipment, or "
        "origin/destination/status/cargo_type/date range to list shipments. Returns the total number of "
        "matches plus up to `limit` rows. Dates filter on planned_departure.",
        QueryShipmentsInput,
        _query_shipments,
    ),
    Tool(
        "get_route_stats",
        "Aggregated metrics for ONE route (origin → destination): shipment counts, average / median / p90 "
        "delay hours, on-time rate (on time = arrived ≤ 24h late, computed over completed shipments). "
        "Optional date range on planned_departure.",
        RouteStatsInput,
        _route_stats,
    ),
    Tool(
        "rank_routes",
        "Rank routes by a metric for a quarter. Use for questions like 'which routes had the highest average "
        "delay this quarter'. period='latest' resolves to the most recent COMPLETE quarter in the data — tell "
        "the user which quarter was used. Small-sample routes are excluded via min_completed.",
        RankRoutesInput,
        _rank_routes,
    ),
    Tool(
        "predict_delay",
        "Delay-risk prediction for an existing shipment from the ML model (probability of arriving > 24h late, "
        "risk band). Uses booking-time information only.",
        PredictDelayInput,
        _predict_delay,
    ),
)


@dataclass
class ToolResult:
    source_id: str | None
    tool: str
    input: dict[str, Any]
    ok: bool
    payload: dict[str, Any]
    latency_ms: float
    truncated: bool = False

    def to_model_content(self) -> str:
        body = {"source_id": self.source_id, "tool": self.tool, "ok": self.ok}
        body["data" if self.ok else "error"] = self.payload
        if self.truncated:
            body["truncated"] = True
        return json.dumps(body, default=str)


@dataclass
class ToolRegistry:
    api: SupplyChainApi
    tools: tuple[Tool, ...] = TOOLS
    max_result_chars: int = 6000
    _counter: int = field(default=0, init=False)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools]

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def execute(self, name: str, raw_input: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        tool = next((t for t in self.tools if t.name == name), None)

        def done(ok: bool, payload: dict[str, Any], source_id: str | None = None) -> ToolResult:
            text = json.dumps(payload, default=str)
            truncated = len(text) > self.max_result_chars
            if truncated and "items" in payload:
                payload = dict(payload)
                while payload["items"] and len(json.dumps(payload, default=str)) > self.max_result_chars:
                    payload["items"] = payload["items"][:-1]
                payload["returned"] = len(payload["items"])
            return ToolResult(
                source_id, name, raw_input, ok, payload, round((time.perf_counter() - start) * 1000, 1), truncated
            )

        if tool is None:
            return done(False, {"code": "UNKNOWN_TOOL", "message": f"No tool named {name!r}"})
        try:
            args = tool.input_model.model_validate(raw_input or {})
        except ValidationError as exc:
            errors = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors()]
            return done(False, {"code": "INVALID_ARGUMENTS", "errors": errors})
        try:
            payload = tool.handler(self.api, args)
        except ApiError as exc:
            return done(False, exc.to_dict())
        self._counter += 1
        return done(True, payload, source_id=f"S{self._counter}")
