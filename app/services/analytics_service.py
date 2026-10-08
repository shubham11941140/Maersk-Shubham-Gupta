"""Ports reference and route rankings."""

from __future__ import annotations

from app.errors import InvalidRequestError
from app.repositories.analytics_repository import AnalyticsRepository
from app.schemas.analytics import (
    PeriodInfo,
    Port,
    PortListResponse,
    RankedRoute,
    RouteRankingResponse,
)

LATEST = "latest"
ALL = "all"


class AnalyticsService:
    def __init__(self, repository: AnalyticsRepository) -> None:
        self._repo = repository

    def ports(self) -> PortListResponse:
        return PortListResponse(items=[Port.model_validate(r) for r in self._repo.list_ports()])

    def route_rankings(
        self,
        metric: str = "avg_delay_hours",
        order: str = "desc",
        period: str = LATEST,
        min_completed: int = 3,
        limit: int = 10,
    ) -> RouteRankingResponse:
        periods = [PeriodInfo.model_validate(p) for p in self._repo.periods()]
        complete = [p.period for p in periods if p.complete]
        latest_complete = complete[-1] if complete else None
        known = {p.period: p for p in periods}

        note = None
        if period == LATEST:
            if latest_complete is None:
                raise InvalidRequestError("No complete quarter in the data", {"available_periods": list(known)})
            resolved = latest_complete
            note = (
                f"'latest' resolves to the most recent COMPLETE quarter in the data ({resolved}). "
                f"Data ends {periods[-1].period}, which is partial."
                if periods and not periods[-1].complete
                else None
            )
        elif period == ALL:
            resolved = ALL
        elif period in known:
            resolved = period
            if not known[period].complete:
                note = f"{period} is only partially covered by the data; treat rankings with caution."
        else:
            raise InvalidRequestError(
                f"Unknown period '{period}'", {"available_periods": list(known), "also_accepted": [LATEST, ALL]}
            )

        rows = self._repo.route_rankings(
            metric, order == "desc", None if resolved == ALL else resolved, min_completed, limit
        )
        return RouteRankingResponse(
            metric=metric,
            order=order,
            period=resolved,
            period_is_complete=True if resolved == ALL else known[resolved].complete,
            latest_complete_period=latest_complete,
            available_periods=periods,
            min_completed=min_completed,
            items=[
                RankedRoute(
                    rank=i,
                    **{k: v for k, v in r.items() if k not in ("avg_delay_hours", "on_time_rate")},
                    avg_delay_hours=None if r["avg_delay_hours"] is None else round(float(r["avg_delay_hours"]), 2),
                    on_time_rate=None if r["on_time_rate"] is None else round(float(r["on_time_rate"]), 4),
                )
                for i, r in enumerate(rows, start=1)
            ],
            note=note,
        )
