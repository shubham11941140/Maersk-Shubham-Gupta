"""Canonical reference values and thresholds used across the platform."""

from __future__ import annotations

from typing import Final

# A shipment is "on time" when it arrives no more than this many hours late.
ON_TIME_THRESHOLD_HOURS: Final[float] = 24.0

DATETIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

CANONICAL_STATUSES: Final[frozenset[str]] = frozenset({"DELIVERED", "DELAYED", "CANCELLED"})
# Statuses that appear in the raw data but are not part of the canonical set.
# They are mapped to a canonical value where the mapping is unambiguous.
STATUS_ALIASES: Final[dict[str, str]] = {
    "DELIVERED": "DELIVERED",
    "COMPLETED": "DELIVERED",
    "COMPLETE": "DELIVERED",
    "DELAYED": "DELAYED",
    "CANCELLED": "CANCELLED",
    "CANCELED": "CANCELLED",
}
# Status assigned in the curated layer when the raw status cannot be trusted.
UNKNOWN_STATUS: Final[str] = "UNKNOWN"

CANONICAL_CARGO_TYPES: Final[tuple[str, ...]] = (
    "Apparel",
    "Automotive Parts",
    "Chemicals",
    "Consumer Goods",
    "Electronics",
    "Food & Beverage",
    "Furniture",
    "Machinery",
    "Pharmaceuticals",
    "Raw Materials",
)
UNKNOWN_CARGO_TYPE: Final[str] = "Unknown"

CANONICAL_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"ARRIVAL", "BERTHED", "CUSTOMS_CLEARED", "DELAYED", "DEPARTURE", "INSPECTION", "LOADING_COMPLETE"}
)

# Plausibility bounds derived from profiling (p99 of container_count is ~50).
MAX_PLAUSIBLE_CONTAINERS: Final[int] = 100
MAX_PLAUSIBLE_EVENT_DELAY_MINUTES: Final[int] = 24 * 60

# Vessel ids that look like test / placeholder values.
SENTINEL_VESSEL_IDS: Final[frozenset[str]] = frozenset({"VSL-0000", "VSL-9999"})

# Known corrections for reference data (documented in the DQ report).
PORT_COUNTRY_FIXES: Final[dict[str, str]] = {"BEANR": "Belgium"}


def canonical_cargo_type(raw: str | None) -> str | None:
    """Map a raw cargo type to its canonical spelling, or None if it is not recognisable."""
    if raw is None:
        return None
    cleaned = " ".join(str(raw).split()).casefold()
    if not cleaned:
        return None
    for canonical in CANONICAL_CARGO_TYPES:
        if canonical.casefold() == cleaned:
            return canonical
    return None


def canonical_status(raw: str | None) -> str | None:
    """Map a raw status to its canonical value, or None if it cannot be mapped."""
    if raw is None:
        return None
    cleaned = str(raw).strip().upper()
    return STATUS_ALIASES.get(cleaned)
