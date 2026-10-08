"""Data-access layer. Services depend on the Protocol, not on DuckDB (repository pattern)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Protocol

from app.db import Database

SORTABLE_COLUMNS = ("planned_departure", "booking_date", "planned_arrival", "shipment_id", "actual_delay_hours")


@dataclass(frozen=True)
class ShipmentFilters:
    origin: str | None = None
    destination: str | None = None
    status: str | None = None
    cargo_type: str | None = None
    date_from: date | None = None  # inclusive, applied to planned_departure
    date_to: date | None = None  # inclusive, applied to planned_departure


@dataclass(frozen=True)
class PageRequest:
    page: int = 1
    page_size: int = 20
    sort_by: str = "planned_departure"
    descending: bool = False

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class ShipmentRepository(Protocol):
    def get_by_id(self, shipment_id: str) -> dict[str, Any] | None: ...

    def list(self, filters: ShipmentFilters, page: PageRequest) -> tuple[list[dict[str, Any]], int]: ...

    def route_stats(
        self, origin: str, destination: str, date_from: date | None, date_to: date | None
    ) -> dict[str, Any]: ...

    def existing_ports(self, codes: list[str]) -> set[str]: ...

    def get_booking_inputs(self, shipment_id: str) -> dict[str, Any] | None: ...


_SHIPMENT_COLUMNS = """
    s.shipment_id, s.customer_id, s.origin_port, s.destination_port, s.route_key, s.vessel_id,
    s.booking_date, s.planned_departure, s.actual_departure, s.planned_arrival, s.actual_arrival,
    s.container_count, s.cargo_type, s.weight_tons, s.status, s.status_raw,
    s.actual_delay_hours, s.on_time_flag, s.transit_days_planned, s.transit_days_actual,
    s.booking_lead_days, s.dq_flags
"""


def _date_bounds(filters: ShipmentFilters) -> tuple[datetime | None, datetime | None]:
    start = datetime.combine(filters.date_from, time.min) if filters.date_from else None
    end = datetime.combine(filters.date_to + timedelta(days=1), time.min) if filters.date_to else None
    return start, end


class DuckDBShipmentRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def get_by_id(self, shipment_id: str) -> dict[str, Any] | None:
        return self._db.fetch_one(
            f"""
            SELECT {_SHIPMENT_COLUMNS},
                   o.port_name AS origin_port_name, o.country AS origin_country, o.region AS origin_region,
                   d.port_name AS destination_port_name, d.country AS destination_country,
                   d.region AS destination_region
            FROM curated.shipments AS s
            JOIN curated.ports AS o ON o.port_code = s.origin_port
            JOIN curated.ports AS d ON d.port_code = s.destination_port
            WHERE s.shipment_id = ?
            """,  # noqa: S608 - column list is a constant
            [shipment_id],
        )

    def list(self, filters: ShipmentFilters, page: PageRequest) -> tuple[list[dict[str, Any]], int]:
        if page.sort_by not in SORTABLE_COLUMNS:  # whitelist — never interpolate user input
            raise ValueError(f"Unsupported sort column: {page.sort_by}")

        clauses, params = ["1 = 1"], []
        for column, value in (
            ("s.origin_port", filters.origin),
            ("s.destination_port", filters.destination),
            ("s.status", filters.status),
            ("s.cargo_type", filters.cargo_type),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        start, end = _date_bounds(filters)
        if start:
            clauses.append("s.planned_departure >= ?")
            params.append(start)
        if end:
            clauses.append("s.planned_departure < ?")
            params.append(end)
        where = " AND ".join(clauses)

        total = self._db.fetch_one(f"SELECT count(*) AS n FROM curated.shipments AS s WHERE {where}", params)["n"]  # noqa: S608
        direction = "DESC" if page.descending else "ASC"
        rows = self._db.fetch_dicts(
            f"""
            SELECT {_SHIPMENT_COLUMNS}
            FROM curated.shipments AS s
            WHERE {where}
            ORDER BY s.{page.sort_by} {direction} NULLS LAST, s.shipment_id
            LIMIT ? OFFSET ?
            """,  # noqa: S608 - sort column whitelisted above
            [*params, page.page_size, page.offset],
        )
        return rows, int(total)

    def route_stats(
        self, origin: str, destination: str, date_from: date | None, date_to: date | None
    ) -> dict[str, Any]:
        start, end = _date_bounds(ShipmentFilters(date_from=date_from, date_to=date_to))
        return (
            self._db.fetch_one(
                """
            SELECT
                count(*)                                                AS shipment_count,
                count(actual_delay_hours)                               AS completed_count,
                count(*) FILTER (WHERE status = 'CANCELLED')            AS cancelled_count,
                count(*) FILTER (WHERE status = 'UNKNOWN')              AS unknown_outcome_count,
                count(*) FILTER (WHERE on_time_flag)                    AS on_time_count,
                avg(actual_delay_hours)                                 AS avg_delay_hours,
                median(actual_delay_hours)                              AS median_delay_hours,
                quantile_cont(actual_delay_hours, 0.9)                  AS p90_delay_hours,
                max(actual_delay_hours)                                 AS max_delay_hours
            FROM curated.shipments
            WHERE origin_port = ? AND destination_port = ?
              AND (CAST(? AS TIMESTAMP) IS NULL OR planned_departure >= CAST(? AS TIMESTAMP))
              AND (CAST(? AS TIMESTAMP) IS NULL OR planned_departure <  CAST(? AS TIMESTAMP))
            """,
                [origin, destination, start, start, end, end],
            )
            or {}
        )

    def existing_ports(self, codes: list[str]) -> set[str]:
        if not codes:
            return set()
        placeholders = ", ".join("?" for _ in codes)
        rows = self._db.fetch_dicts(
            f"SELECT port_code FROM curated.ports WHERE port_code IN ({placeholders})",  # noqa: S608
            codes,
        )
        return {r["port_code"] for r in rows}

    def get_booking_inputs(self, shipment_id: str) -> dict[str, Any] | None:
        """Only columns known at booking time — the leakage boundary for the model."""
        return self._db.fetch_one(
            """
            SELECT shipment_id, origin_port, destination_port, cargo_type, container_count, weight_tons,
                   booking_date, planned_departure, planned_arrival
            FROM curated.shipments WHERE shipment_id = ?
            """,
            [shipment_id],
        )
