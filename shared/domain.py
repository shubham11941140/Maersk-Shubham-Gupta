"""Pure domain functions — no I/O, trivially unit-testable."""

from __future__ import annotations

from datetime import datetime

from shared.reference import ON_TIME_THRESHOLD_HOURS

_SECONDS_PER_HOUR = 3600.0
_SECONDS_PER_DAY = 86400.0


def delay_hours(planned_arrival: datetime | None, actual_arrival: datetime | None) -> float | None:
    """Hours between planned and actual arrival (positive = late). Null-safe."""
    if planned_arrival is None or actual_arrival is None:
        return None
    return (actual_arrival - planned_arrival).total_seconds() / _SECONDS_PER_HOUR


def is_on_time(delay: float | None, threshold_hours: float = ON_TIME_THRESHOLD_HOURS) -> bool | None:
    """True when the shipment arrived at most ``threshold_hours`` late. Null-safe."""
    if delay is None:
        return None
    return delay <= threshold_hours


def route_key(origin_port: str, destination_port: str) -> str:
    """Human-readable route identifier, e.g. ``CNSHA → NLRTM``."""
    return f"{origin_port} → {destination_port}"


def transit_days(start: datetime | None, end: datetime | None) -> float | None:
    """Days between two timestamps. Null-safe."""
    if start is None or end is None:
        return None
    return (end - start).total_seconds() / _SECONDS_PER_DAY
