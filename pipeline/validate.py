"""Post-build data contract. Runs on the freshly built warehouse BEFORE it is
swapped into place: if any rule fails, the build is rejected and the previous
warehouse keeps serving.

Each rule is a SQL query returning the number of violating rows (0 = pass).
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from shared.reference import (
    CANONICAL_CARGO_TYPES,
    CANONICAL_STATUSES,
    MAX_PLAUSIBLE_CONTAINERS,
    ON_TIME_THRESHOLD_HOURS,
    UNKNOWN_CARGO_TYPE,
    UNKNOWN_STATUS,
)

# Guard against a rule change that suddenly quarantines most of the data.
MAX_QUARANTINE_PCT = 10.0


class PipelineValidationError(RuntimeError):
    def __init__(self, failures: list[ValidationResult]) -> None:
        self.failures = failures
        lines = "\n".join(f"  - {f.rule}: {f.violations} violation(s)" for f in failures)
        super().__init__(f"Warehouse failed {len(failures)} data-contract rule(s):\n{lines}")


@dataclass(frozen=True)
class ValidationResult:
    rule: str
    description: str
    violations: int

    @property
    def passed(self) -> bool:
        return self.violations == 0


def _in_list(values) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in sorted(values))


_STATUSES = _in_list({*CANONICAL_STATUSES, UNKNOWN_STATUS})
_CARGO = _in_list({*CANONICAL_CARGO_TYPES, UNKNOWN_CARGO_TYPE})
_RAW_SHIP_COLS = (
    "shipment_id, customer_id, origin_port, destination_port, vessel_id, booking_date, "
    "planned_departure, actual_departure, planned_arrival, actual_arrival, container_count, "
    "cargo_type, weight_tons, status"
)
_RAW_EVENT_COLS = "event_id, event_timestamp, port_code, vessel_id, event_type, delay_minutes, notes"

RULES: list[tuple[str, str, str]] = [
    # --- keys -----------------------------------------------------------------
    (
        "pk_shipments_unique",
        "shipment_id is unique and not null in curated.shipments",
        "SELECT count(*) - count(DISTINCT shipment_id) + count(*) FILTER (WHERE shipment_id IS NULL) "
        "FROM curated.shipments",
    ),
    (
        "pk_ports_unique",
        "port_code is unique in curated.ports",
        "SELECT count(*) - count(DISTINCT port_code) FROM curated.ports",
    ),
    (
        "pk_events_unique",
        "event_key is unique in curated.port_events",
        "SELECT count(*) - count(DISTINCT event_key) FROM curated.port_events",
    ),
    # --- referential integrity ------------------------------------------------
    (
        "fk_shipment_ports",
        "every shipment origin/destination exists in curated.ports",
        "SELECT count(*) FROM curated.shipments s WHERE s.origin_port NOT IN (SELECT port_code FROM curated.ports) "
        "OR s.destination_port NOT IN (SELECT port_code FROM curated.ports)",
    ),
    (
        "fk_event_ports",
        "every event port exists in curated.ports",
        "SELECT count(*) FROM curated.port_events WHERE port_code NOT IN (SELECT port_code FROM curated.ports)",
    ),
    # --- completeness ---------------------------------------------------------
    (
        "required_fields_not_null",
        "planned timestamps, route_key, status, cargo_type are never null",
        "SELECT count(*) FROM curated.shipments WHERE booking_date IS NULL OR planned_departure IS NULL "
        "OR planned_arrival IS NULL OR route_key IS NULL OR status IS NULL OR cargo_type IS NULL",
    ),
    (
        "ports_complete",
        "no blank country / region / congestion in curated.ports",
        "SELECT count(*) FROM curated.ports WHERE country IS NULL OR region IS NULL OR avg_congestion_score IS NULL",
    ),
    # --- domains --------------------------------------------------------------
    (
        "status_domain",
        "status is canonical (or UNKNOWN)",
        f"SELECT count(*) FROM curated.shipments WHERE status NOT IN ({_STATUSES})",
    ),
    (
        "cargo_type_domain",
        "cargo_type is canonical (or Unknown)",
        f"SELECT count(*) FROM curated.shipments WHERE cargo_type NOT IN ({_CARGO})",
    ),
    (
        "container_count_range",
        f"container_count is NULL or 1..{MAX_PLAUSIBLE_CONTAINERS}",
        f"SELECT count(*) FROM curated.shipments WHERE container_count NOT BETWEEN 1 AND {MAX_PLAUSIBLE_CONTAINERS}",
    ),
    ("weight_positive", "weight_tons is NULL or > 0", "SELECT count(*) FROM curated.shipments WHERE weight_tons <= 0"),
    # --- derived-field correctness --------------------------------------------
    (
        "derived_delay_hours",
        "actual_delay_hours = actual_arrival − planned_arrival (hours)",
        "SELECT count(*) FROM curated.shipments WHERE actual_delay_hours IS DISTINCT FROM "
        "date_diff('minute', planned_arrival, actual_arrival) / 60.0",
    ),
    (
        "derived_on_time_flag",
        f"on_time_flag = actual_delay_hours ≤ {ON_TIME_THRESHOLD_HOURS:g} (null-safe)",
        f"SELECT count(*) FROM curated.shipments "
        f"WHERE on_time_flag IS DISTINCT FROM (actual_delay_hours <= {ON_TIME_THRESHOLD_HOURS})",
    ),
    (
        "derived_route_key",
        "route_key = origin_port || ' → ' || destination_port",
        "SELECT count(*) FROM curated.shipments WHERE route_key <> origin_port || ' → ' || destination_port",
    ),
    (
        "derived_transit_days",
        "transit_days_planned > 0; transit_days_actual NULL or > 0",
        "SELECT count(*) FROM curated.shipments WHERE transit_days_planned <= 0 OR transit_days_actual <= 0",
    ),
    (
        "status_consistent_with_delay",
        "DELAYED ⇔ delay > 24h, DELIVERED ⇔ ≤ 24h, others have no delay",
        f"SELECT count(*) FROM curated.shipments WHERE "
        f"(status = 'DELAYED' AND NOT actual_delay_hours > {ON_TIME_THRESHOLD_HOURS}) OR "
        f"(status = 'DELIVERED' AND NOT actual_delay_hours <= {ON_TIME_THRESHOLD_HOURS}) OR "
        f"(status IN ('CANCELLED', 'UNKNOWN') AND actual_delay_hours IS NOT NULL) OR "
        f"(status IN ('DELAYED', 'DELIVERED') AND actual_delay_hours IS NULL)",
    ),
    # --- reconciliation: no row silently lost or invented ---------------------
    (
        "reconcile_shipments",
        "raw = exact duplicates dropped + curated + quarantined",
        f"SELECT abs((SELECT count(*) FROM raw.shipments) - "
        f"((SELECT count(*) FROM raw.shipments) - (SELECT count(*) FROM (SELECT DISTINCT {_RAW_SHIP_COLS} "
        f"FROM raw.shipments))) - (SELECT count(*) FROM curated.shipments) - "
        f"(SELECT count(*) FROM quarantine.shipments))",
    ),
    (
        "reconcile_port_events",
        "raw = exact duplicates dropped + curated + quarantined",
        f"SELECT abs((SELECT count(*) FROM raw.port_events) - "
        f"((SELECT count(*) FROM raw.port_events) - (SELECT count(*) FROM (SELECT DISTINCT {_RAW_EVENT_COLS} "
        f"FROM raw.port_events))) - (SELECT count(*) FROM curated.port_events) - "
        f"(SELECT count(*) FROM quarantine.port_events))",
    ),
    (
        "curated_not_empty",
        "curated.shipments has rows",
        "SELECT CASE WHEN (SELECT count(*) FROM curated.shipments) = 0 THEN 1 ELSE 0 END",
    ),
]


def _quarantine_rule(max_pct: float) -> tuple[str, str, str]:
    return (
        "quarantine_ratio",
        f"quarantined shipments ≤ {max_pct:g}% of raw",
        f"SELECT CASE WHEN (SELECT count(*) FROM quarantine.shipments) * 100.0 / "
        f"greatest((SELECT count(*) FROM raw.shipments), 1) > {float(max_pct)} "
        f"THEN (SELECT count(*) FROM quarantine.shipments) ELSE 0 END",
    )


def validate(con: duckdb.DuckDBPyConnection, max_quarantine_pct: float = MAX_QUARANTINE_PCT) -> list[ValidationResult]:
    return [
        ValidationResult(rule=name, description=desc, violations=int(con.execute(sql).fetchone()[0]))
        for name, desc, sql in [*RULES, _quarantine_rule(max_quarantine_pct)]
    ]


def assert_valid(
    con: duckdb.DuckDBPyConnection, max_quarantine_pct: float = MAX_QUARANTINE_PCT
) -> list[ValidationResult]:
    results = validate(con, max_quarantine_pct)
    failures = [r for r in results if not r.passed]
    if failures:
        raise PipelineValidationError(failures)
    return results
