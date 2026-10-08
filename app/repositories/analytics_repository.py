"""Read-side analytics over the serving layer (ports reference, route rankings).

Kept separate from ShipmentRepository (interface segregation): consumers that only need
rankings — like the GenAI assistant's tools — depend on this small Protocol.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.db import Database

RANKING_METRICS = ("avg_delay_hours", "on_time_rate", "completed_count", "shipment_count")


class AnalyticsRepository(Protocol):
    def list_ports(self) -> list[dict[str, Any]]: ...

    def periods(self) -> list[dict[str, Any]]: ...

    def route_rankings(
        self, metric: str, descending: bool, period: str | None, min_completed: int, limit: int
    ) -> list[dict[str, Any]]: ...


class DuckDBAnalyticsRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def list_ports(self) -> list[dict[str, Any]]:
        return self._db.fetch_dicts(
            "SELECT port_code, port_name, country, region, timezone, avg_congestion_score "
            "FROM curated.ports ORDER BY port_code"
        )

    def periods(self) -> list[dict[str, Any]]:
        """Quarters with data, flagged complete only if the data covers the whole quarter."""
        return self._db.fetch_dicts(
            """
            WITH bounds AS (
                SELECT date_trunc('day', min(planned_departure)) AS lo, max(planned_departure) AS hi
                FROM curated.shipments
            ),
            q AS (
                SELECT year(planned_departure) || '-Q' || quarter(planned_departure) AS period,
                       min(date_trunc('quarter', planned_departure))                  AS quarter_start,
                       count(*)                                                       AS shipments,
                       count(actual_delay_hours)                                      AS completed
                FROM curated.shipments
                GROUP BY 1
            )
            SELECT q.period, q.quarter_start, q.shipments, q.completed,
                   (q.quarter_start >= b.lo AND q.quarter_start + INTERVAL 3 MONTH <= b.hi) AS complete
            FROM q, bounds AS b
            ORDER BY q.quarter_start
            """
        )

    def route_rankings(
        self, metric: str, descending: bool, period: str | None, min_completed: int, limit: int
    ) -> list[dict[str, Any]]:
        if metric not in RANKING_METRICS:  # whitelist: the column name is interpolated
            raise ValueError(f"Unsupported ranking metric: {metric}")
        direction = "DESC" if descending else "ASC"
        source = "serving.route_quarterly_stats WHERE period = ? AND" if period else "serving.route_stats WHERE"
        params: list[Any] = [period] if period else []
        return self._db.fetch_dicts(
            f"""
            SELECT route_key, origin_port, destination_port, shipment_count, completed_count,
                   avg_delay_hours, on_time_rate
            FROM {source} completed_count >= ?
            ORDER BY {metric} {direction} NULLS LAST, completed_count DESC, route_key
            LIMIT ?
            """,  # noqa: S608 - metric whitelisted, source chosen from two constants
            [*params, min_completed, limit],
        )
