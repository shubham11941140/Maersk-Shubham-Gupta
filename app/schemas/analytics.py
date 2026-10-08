from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RankingMetric = Literal["avg_delay_hours", "on_time_rate", "completed_count", "shipment_count"]


class Port(BaseModel):
    port_code: str
    port_name: str
    country: str | None
    region: str
    timezone: str
    avg_congestion_score: float


class PortListResponse(BaseModel):
    items: list[Port]


class PeriodInfo(BaseModel):
    period: str = Field(examples=["2025-Q3"])
    complete: bool = Field(description="False when the data covers only part of the quarter")
    shipments: int
    completed: int


class RankedRoute(BaseModel):
    rank: int
    route_key: str
    origin_port: str
    destination_port: str
    shipment_count: int
    completed_count: int
    avg_delay_hours: float | None
    on_time_rate: float | None


class RouteRankingResponse(BaseModel):
    metric: RankingMetric
    order: Literal["asc", "desc"]
    period: str = Field(description="Quarter used, or 'all'")
    period_is_complete: bool
    latest_complete_period: str | None
    available_periods: list[PeriodInfo]
    min_completed: int = Field(description="Routes with fewer completed shipments are excluded (small samples)")
    items: list[RankedRoute]
    note: str | None = None
