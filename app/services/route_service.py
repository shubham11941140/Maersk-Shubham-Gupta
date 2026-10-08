"""Aggregated route metrics."""

from __future__ import annotations

from datetime import date

from app.errors import InvalidRequestError, PortNotFoundError
from app.repositories.shipment_repository import ShipmentRepository
from app.schemas.routes import RouteStatsResponse
from shared.domain import route_key
from shared.reference import ON_TIME_THRESHOLD_HOURS


def _round(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(float(value), digits)


class RouteService:
    def __init__(self, repository: ShipmentRepository) -> None:
        self._repo = repository

    def stats(
        self, origin: str, destination: str, date_from: date | None = None, date_to: date | None = None
    ) -> RouteStatsResponse:
        if origin == destination:
            raise InvalidRequestError("origin and destination must differ", {"origin": origin})
        if date_from and date_to and date_from > date_to:
            raise InvalidRequestError("date_from must be on or before date_to")

        missing = sorted({origin, destination} - self._repo.existing_ports([origin, destination]))
        if missing:
            raise PortNotFoundError(missing)

        row = self._repo.route_stats(origin, destination, date_from, date_to)
        completed = int(row.get("completed_count") or 0)
        on_time = int(row.get("on_time_count") or 0)
        return RouteStatsResponse(
            origin_port=origin,
            destination_port=destination,
            route_key=route_key(origin, destination),
            date_from=date_from,
            date_to=date_to,
            shipment_count=int(row.get("shipment_count") or 0),
            completed_count=completed,
            cancelled_count=int(row.get("cancelled_count") or 0),
            unknown_outcome_count=int(row.get("unknown_outcome_count") or 0),
            avg_delay_hours=_round(row.get("avg_delay_hours")),
            median_delay_hours=_round(row.get("median_delay_hours")),
            p90_delay_hours=_round(row.get("p90_delay_hours")),
            max_delay_hours=_round(row.get("max_delay_hours")),
            on_time_rate=round(on_time / completed, 4) if completed else None,
            on_time_threshold_hours=ON_TIME_THRESHOLD_HOURS,
        )
