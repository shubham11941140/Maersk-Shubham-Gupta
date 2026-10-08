"""The DQ check catalogue.

Every check is a small pure function registered with ``@register``. It receives
the raw data (all columns as strings, exactly as on disk) and returns a boolean
mask over the rows of its dataset: True = row is affected.

Adding a check means writing one function — the runner, the JSON report, the
API endpoint and the CI baseline gate pick it up automatically (registry pattern).
"""

from __future__ import annotations

import pandas as pd

from dq.models import Action, Check, CheckFn, RawData, Severity
from shared.reference import (
    CANONICAL_CARGO_TYPES,
    CANONICAL_EVENT_TYPES,
    CANONICAL_STATUSES,
    DATETIME_FORMAT,
    MAX_PLAUSIBLE_CONTAINERS,
    MAX_PLAUSIBLE_EVENT_DELAY_MINUTES,
    ON_TIME_THRESHOLD_HOURS,
    SENTINEL_VESSEL_IDS,
    canonical_status,
)

REGISTRY: list[Check] = []

KEY_COLUMNS = {"ports": "port_code", "shipments": "shipment_id", "port_events": "event_id"}
_SHIPMENT_TS_COLUMNS = ("booking_date", "planned_departure", "actual_departure", "planned_arrival", "actual_arrival")
_REQUIRED_SHIPMENT_TS = ("booking_date", "planned_departure", "planned_arrival")
_MIN_ROUTE_SAMPLE = 5
_TRANSIT_DEVIATION = 0.5


def register(
    name: str,
    dataset: str,
    *,
    description: str,
    detection: str,
    severity: Severity,
    action: Action,
    rationale: str,
):
    """Decorator that adds a check function to the registry."""

    def decorator(fn: CheckFn) -> CheckFn:
        if any(c.name == name for c in REGISTRY):
            raise ValueError(f"Duplicate DQ check name: {name}")
        REGISTRY.append(
            Check(
                name=name,
                dataset=dataset,
                key_column=KEY_COLUMNS[dataset],
                description=description,
                detection=detection,
                severity=severity,
                action=action,
                rationale=rationale,
                fn=fn,
            )
        )
        return fn

    return decorator


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _blank(series: pd.Series) -> pd.Series:
    """True where a raw string value is empty / whitespace."""
    return series.fillna("").astype(str).str.strip() == ""


def _ts(series: pd.Series) -> pd.Series:
    """Parse timestamps strictly; blanks and bad formats become NaT."""
    return pd.to_datetime(series.where(~_blank(series)), format=DATETIME_FORMAT, errors="coerce")


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.where(~_blank(series)), errors="coerce")


def _known_ports(data: RawData) -> set[str]:
    return set(data.ports["port_code"].str.strip())


def _delay_hours(s: pd.DataFrame) -> pd.Series:
    return (_ts(s["actual_arrival"]) - _ts(s["planned_arrival"])).dt.total_seconds() / 3600


# =========================================================================== #
# shipments.csv — identity & referential integrity
# =========================================================================== #
@register(
    "shipments.exact_duplicate_rows",
    "shipments",
    description="Rows that are byte-for-byte copies of an earlier row.",
    detection="DataFrame.duplicated() across all 14 columns; the first occurrence is kept.",
    severity=Severity.CRITICAL,
    action=Action.DROP,
    rationale="Pure duplicates double-count shipments in route metrics; the copy carries no extra information.",
)
def exact_duplicate_shipments(data: RawData) -> pd.Series:
    return data.shipments.duplicated(keep="first")


@register(
    "shipments.conflicting_duplicate_ids",
    "shipments",
    description="shipment_id appears more than once with differing field values.",
    detection="After removing exact duplicates, shipment_id values that still occur more than once. "
    "All rows sharing such an id are counted.",
    severity=Severity.CRITICAL,
    action=Action.QUARANTINE,
    rationale="The primary key must be unique. Keep the most complete version (fewest blank fields, then first "
    "seen) and quarantine the others so the conflict can be investigated with the source system.",
)
def conflicting_duplicate_ids(data: RawData) -> pd.Series:
    df = data.shipments.drop_duplicates(keep="first")
    dup_ids = df.loc[df["shipment_id"].duplicated(keep=False), "shipment_id"]
    not_exact = ~data.shipments.duplicated(keep="first")
    return data.shipments["shipment_id"].isin(dup_ids) & not_exact


@register(
    "shipments.invalid_id_format",
    "shipments",
    description="shipment_id / customer_id / vessel_id do not match SHP-#####, CUST-####, VSL-####.",
    detection="Regex full-match on each identifier column.",
    severity=Severity.CRITICAL,
    action=Action.QUARANTINE,
    rationale="Malformed identifiers cannot be joined or looked up reliably.",
)
def invalid_shipment_ids(data: RawData) -> pd.Series:
    s = data.shipments
    return (
        ~s["shipment_id"].str.fullmatch(r"SHP-\d{5}")
        | ~s["customer_id"].str.fullmatch(r"CUST-\d{4}")
        | ~s["vessel_id"].str.fullmatch(r"VSL-\d{4}")
    )


@register(
    "shipments.unknown_port_code",
    "shipments",
    description="origin_port or destination_port is not in ports.csv (e.g. XXTST, ZZZZZ, UNKNW).",
    detection="Anti-join of origin_port / destination_port against ports.port_code.",
    severity=Severity.CRITICAL,
    action=Action.QUARANTINE,
    rationale="Placeholder / test port codes cannot be attributed to a real route; including them would create "
    "phantom routes in the stats endpoint and noise in the model.",
)
def unknown_port_code(data: RawData) -> pd.Series:
    known = _known_ports(data)
    s = data.shipments
    return ~s["origin_port"].str.strip().isin(known) | ~s["destination_port"].str.strip().isin(known)


@register(
    "shipments.same_origin_destination",
    "shipments",
    description="origin_port equals destination_port.",
    detection="Row-wise equality of the two port columns.",
    severity=Severity.WARNING,
    action=Action.QUARANTINE,
    rationale="A shipment cannot start and end at the same port in this domain.",
)
def same_origin_destination(data: RawData) -> pd.Series:
    s = data.shipments
    return s["origin_port"].str.strip() == s["destination_port"].str.strip()


# =========================================================================== #
# shipments.csv — timestamps
# =========================================================================== #
@register(
    "shipments.unparseable_timestamp",
    "shipments",
    description="A non-blank timestamp column does not parse as YYYY-MM-DD HH:MM:SS.",
    detection="Strict pd.to_datetime(format=...) per timestamp column; non-blank values that become NaT.",
    severity=Severity.CRITICAL,
    action=Action.QUARANTINE,
    rationale="Delay and transit calculations depend on every timestamp being valid.",
)
def unparseable_shipment_timestamp(data: RawData) -> pd.Series:
    s = data.shipments
    mask = pd.Series(False, index=s.index)
    for col in _SHIPMENT_TS_COLUMNS:
        mask |= ~_blank(s[col]) & _ts(s[col]).isna()
    return mask


@register(
    "shipments.missing_planned_timestamp",
    "shipments",
    description="booking_date, planned_departure or planned_arrival is blank.",
    detection="Blank check on the three planned / booking columns.",
    severity=Severity.CRITICAL,
    action=Action.QUARANTINE,
    rationale="Planned dates are required for delay calculation and are core model features.",
)
def missing_planned_timestamp(data: RawData) -> pd.Series:
    s = data.shipments
    mask = pd.Series(False, index=s.index)
    for col in _REQUIRED_SHIPMENT_TS:
        mask |= _blank(s[col])
    return mask


@register(
    "shipments.planned_arrival_not_after_departure",
    "shipments",
    description="planned_arrival is on or before planned_departure, or planned_departure is before booking_date.",
    detection="Pairwise comparison of parsed booking_date ≤ planned_departure < planned_arrival.",
    severity=Severity.CRITICAL,
    action=Action.QUARANTINE,
    rationale="An impossible plan makes planned transit time and the delay baseline meaningless.",
)
def planned_sequence_invalid(data: RawData) -> pd.Series:
    s = data.shipments
    booking, p_dep, p_arr = _ts(s["booking_date"]), _ts(s["planned_departure"]), _ts(s["planned_arrival"])
    return (p_arr <= p_dep) | (p_dep < booking)


@register(
    "shipments.actual_departure_before_booking",
    "shipments",
    description="actual_departure is earlier than booking_date — the vessel left before the shipment was booked.",
    detection="Parsed actual_departure < booking_date. Found by profiling actual − planned departure: every "
    "early departure beyond 4h is one of these rows (up to 16 days early).",
    severity=Severity.CRITICAL,
    action=Action.FIX,
    rationale="The actual departure timestamp is impossible, so it is set to NULL (transit_days_actual becomes "
    "NULL) while actual_arrival — which looks plausible — is kept so the delay metric survives. Row is flagged.",
)
def actual_departure_before_booking(data: RawData) -> pd.Series:
    s = data.shipments
    return _ts(s["actual_departure"]) < _ts(s["booking_date"])


@register(
    "shipments.actual_arrival_before_departure",
    "shipments",
    description="actual_arrival is on or before actual_departure.",
    detection="Parsed actual_arrival ≤ actual_departure.",
    severity=Severity.CRITICAL,
    action=Action.FIX,
    rationale="Impossible sequence; actual timestamps would be nulled and the row flagged.",
)
def actual_arrival_before_departure(data: RawData) -> pd.Series:
    s = data.shipments
    return _ts(s["actual_arrival"]) <= _ts(s["actual_departure"])


@register(
    "shipments.planned_transit_outlier_for_route",
    "shipments",
    description=f"Planned transit time deviates more than {int(_TRANSIT_DEVIATION * 100)}% from the median of "
    f"its route (routes with ≥ {_MIN_ROUTE_SAMPLE} shipments).",
    detection="Group by (origin, destination), compare (planned_arrival − planned_departure) with the route median.",
    severity=Severity.INFO,
    action=Action.NONE,
    rationale="Same-lane transit times vary from 3 to 35 days, which real liner schedules do not. Values are not "
    "provably wrong, so nothing is changed — but it explains why planned transit carries little ML signal.",
)
def planned_transit_outlier_for_route(data: RawData) -> pd.Series:
    s = data.shipments
    transit = (_ts(s["planned_arrival"]) - _ts(s["planned_departure"])).dt.total_seconds() / 86400
    keys = [s["origin_port"].str.strip(), s["destination_port"].str.strip()]
    median = transit.groupby(keys).transform("median")
    size = transit.groupby(keys).transform("count")
    return (size >= _MIN_ROUTE_SAMPLE) & ((transit - median).abs() / median > _TRANSIT_DEVIATION)


# =========================================================================== #
# shipments.csv — status
# =========================================================================== #
@register(
    "shipments.status_non_canonical",
    "shipments",
    description=f"status is not one of {sorted(CANONICAL_STATUSES)} as written (e.g. 'delivered', 'Complete').",
    detection="value_counts() on status; anything outside the canonical set that is not blank / 'N/A'.",
    severity=Severity.WARNING,
    action=Action.FIX,
    rationale="Case / synonym variants are normalised via an alias map. In the curated layer status is then "
    "re-derived from the actual timestamps so it can never disagree with the delay metric.",
)
def status_non_canonical(data: RawData) -> pd.Series:
    st = data.shipments["status"].fillna("")
    return ~_blank(st) & ~st.isin(CANONICAL_STATUSES) & (st.str.strip().str.upper() != "N/A")


@register(
    "shipments.status_missing_or_placeholder",
    "shipments",
    description="status is blank or a placeholder such as 'N/A'.",
    detection="Blank check plus a placeholder list on status.",
    severity=Severity.WARNING,
    action=Action.FIX,
    rationale="Status is re-derived from actual timestamps where possible; otherwise set to UNKNOWN and flagged.",
)
def status_missing(data: RawData) -> pd.Series:
    st = data.shipments["status"].fillna("")
    return _blank(st) | (st.str.strip().str.upper() == "N/A")


@register(
    "shipments.status_actuals_mismatch",
    "shipments",
    description="Status contradicts the actual timestamps: a non-cancelled shipment with no actual_arrival "
    "(e.g. 'COMPLETED' but never arrived), or a CANCELLED shipment that has an actual_arrival.",
    detection="Cross-tab of canonicalised status vs actual_arrival IS NULL. Every non-canonical / blank status "
    "row turned out to have no actuals.",
    severity=Severity.WARNING,
    action=Action.FLAG,
    rationale="These rows have no observed outcome. They are kept (status=UNKNOWN, delay NULL) so the shipment "
    "can still be looked up, but are excluded from on-time metrics and from model training.",
)
def status_actuals_mismatch(data: RawData) -> pd.Series:
    s = data.shipments
    status = s["status"].map(canonical_status)
    has_arrival = ~_blank(s["actual_arrival"])
    return ((status != "CANCELLED") & ~has_arrival) | ((status == "CANCELLED") & has_arrival)


@register(
    "shipments.status_delay_contradiction",
    "shipments",
    description=f"status DELAYED but arrived ≤ {ON_TIME_THRESHOLD_HOURS:.0f}h late, or DELIVERED but arrived "
    f"> {ON_TIME_THRESHOLD_HOURS:.0f}h late.",
    detection="Compare the raw status label with actual_arrival − planned_arrival.",
    severity=Severity.WARNING,
    action=Action.FIX,
    rationale="The curated status is derived from the timestamps, so a contradicting label is overwritten. "
    "Currently passing — kept as a regression guard on the source system's status logic.",
)
def status_delay_contradiction(data: RawData) -> pd.Series:
    s = data.shipments
    delay, status = _delay_hours(s), s["status"].str.strip()
    return ((status == "DELAYED") & (delay <= ON_TIME_THRESHOLD_HOURS)) | (
        (status == "DELIVERED") & (delay > ON_TIME_THRESHOLD_HOURS)
    )


# =========================================================================== #
# shipments.csv — attributes
# =========================================================================== #
@register(
    "shipments.cargo_type_non_canonical",
    "shipments",
    description="cargo_type differs from the canonical spelling only by case / whitespace ('FURNITURE', ' Chemicals').",
    detection="value_counts() showed 18 distinct values for 10 real categories; compared against the canonical list.",
    severity=Severity.WARNING,
    action=Action.FIX,
    rationale="Unnormalised categories split one class into several, hurting both aggregates and one-hot features.",
)
def cargo_type_non_canonical(data: RawData) -> pd.Series:
    ct = data.shipments["cargo_type"].fillna("")
    return ~_blank(ct) & ~ct.isin(CANONICAL_CARGO_TYPES)


@register(
    "shipments.cargo_type_missing",
    "shipments",
    description="cargo_type is blank.",
    detection="Blank check on cargo_type.",
    severity=Severity.WARNING,
    action=Action.FLAG,
    rationale="Kept with cargo_type='Unknown' — the row is otherwise valid and cargo type is not a key field.",
)
def cargo_type_missing(data: RawData) -> pd.Series:
    return _blank(data.shipments["cargo_type"])


@register(
    "shipments.weight_missing",
    "shipments",
    description="weight_tons is blank.",
    detection="Blank check on weight_tons.",
    severity=Severity.WARNING,
    action=Action.FLAG,
    rationale="Kept as NULL; the model imputes with the training median.",
)
def weight_missing(data: RawData) -> pd.Series:
    return _blank(data.shipments["weight_tons"])


@register(
    "shipments.weight_non_positive",
    "shipments",
    description="weight_tons is zero or negative.",
    detection="describe() on weight_tons showed min = −197.87.",
    severity=Severity.WARNING,
    action=Action.FIX,
    rationale="Physically impossible. Set to NULL and flagged rather than dropping an otherwise valid shipment; "
    "taking abs() would be a guess.",
)
def weight_non_positive(data: RawData) -> pd.Series:
    return _num(data.shipments["weight_tons"]) <= 0


@register(
    "shipments.container_count_zero",
    "shipments",
    description="container_count is 0 (or negative).",
    detection="describe() on container_count showed min = 0.",
    severity=Severity.WARNING,
    action=Action.FIX,
    rationale="A booked shipment must carry at least one container. Set to NULL and flagged.",
)
def container_count_zero(data: RawData) -> pd.Series:
    return _num(data.shipments["container_count"]) <= 0


@register(
    "shipments.container_count_outlier",
    "shipments",
    description=f"container_count exceeds {MAX_PLAUSIBLE_CONTAINERS} (bulk of data is 1-50; outliers are 500-1000).",
    detection="Distribution: p75 = 30, max = 996, and nothing between 60 and 500 — a clear gap, not a long tail.",
    severity=Severity.WARNING,
    action=Action.FIX,
    rationale="Values an order of magnitude above the distribution look like unit / keying errors. Set to NULL "
    "and flagged so they don't dominate the model's numeric features.",
)
def container_count_outlier(data: RawData) -> pd.Series:
    return _num(data.shipments["container_count"]) > MAX_PLAUSIBLE_CONTAINERS


# =========================================================================== #
# ports.csv
# =========================================================================== #
@register(
    "ports.missing_field",
    "ports",
    description="A reference field (name, country, region, timezone, congestion) is blank.",
    detection="Blank check on every non-key column.",
    severity=Severity.WARNING,
    action=Action.FIX,
    rationale="Reference data is small and curated by hand: the missing country for BEANR (Antwerp) is filled with "
    "'Belgium' from a documented fix list rather than dropping a port that 180+ shipments use.",
)
def ports_missing_field(data: RawData) -> pd.Series:
    p = data.ports
    mask = pd.Series(False, index=p.index)
    for col in ("port_name", "country", "region", "timezone", "avg_congestion_score"):
        mask |= _blank(p[col])
    return mask


@register(
    "ports.duplicate_port_code",
    "ports",
    description="port_code appears more than once.",
    detection="duplicated() on port_code.",
    severity=Severity.CRITICAL,
    action=Action.DROP,
    rationale="Duplicate reference keys would fan out every join.",
)
def duplicate_port_code(data: RawData) -> pd.Series:
    return data.ports["port_code"].duplicated(keep="first")


@register(
    "ports.congestion_out_of_range",
    "ports",
    description="avg_congestion_score is outside [0, 1] or not numeric.",
    detection="Numeric parse + range check.",
    severity=Severity.WARNING,
    action=Action.FLAG,
    rationale="Score is used as a model feature and should be a normalised value.",
)
def congestion_out_of_range(data: RawData) -> pd.Series:
    raw = data.ports["avg_congestion_score"]
    score = _num(raw)
    return ~_blank(raw) & (score.isna() | (score < 0) | (score > 1))


@register(
    "ports.invalid_timezone_format",
    "ports",
    description="timezone is not of the form UTC±H or UTC±H:MM.",
    detection="Regex full-match on timezone.",
    severity=Severity.WARNING,
    action=Action.FLAG,
    rationale="Needed if timestamps are ever localised; malformed offsets would silently shift times.",
)
def invalid_timezone_format(data: RawData) -> pd.Series:
    tz = data.ports["timezone"].fillna("")
    return ~_blank(tz) & ~tz.str.strip().str.fullmatch(r"UTC[+-]\d{1,2}(:\d{2})?")


# =========================================================================== #
# port_events.csv (streaming-style feed)
# =========================================================================== #
@register(
    "port_events.exact_duplicate_rows",
    "port_events",
    description="Rows that are exact copies of an earlier row (typical at-least-once delivery in a stream).",
    detection="duplicated() across all columns.",
    severity=Severity.WARNING,
    action=Action.DROP,
    rationale="Replayed events would double-count delay minutes.",
)
def exact_duplicate_events(data: RawData) -> pd.Series:
    return data.port_events.duplicated(keep="first")


@register(
    "port_events.duplicate_event_id",
    "port_events",
    description="event_id is reused by events with different content (different port, vessel, timestamp).",
    detection="duplicated(keep=False) on event_id after removing exact duplicates; inspected pairs differ in "
    "every other field.",
    severity=Severity.WARNING,
    action=Action.FIX,
    rationale="These are distinct events with a colliding ID, not replays. Both are kept and the curated layer "
    "uses a deterministic surrogate key (hash of all fields) so neither is lost.",
)
def duplicate_event_id(data: RawData) -> pd.Series:
    e = data.port_events.drop_duplicates(keep="first")
    dup_ids = e.loc[e["event_id"].duplicated(keep=False), "event_id"]
    return data.port_events["event_id"].isin(dup_ids)


@register(
    "port_events.missing_event_type",
    "port_events",
    description="event_type is blank.",
    detection="Blank check on event_type.",
    severity=Severity.WARNING,
    action=Action.QUARANTINE,
    rationale="An event without a type cannot be interpreted; quarantined for replay once the producer is fixed.",
)
def missing_event_type(data: RawData) -> pd.Series:
    return _blank(data.port_events["event_type"])


@register(
    "port_events.invalid_event_type",
    "port_events",
    description=f"event_type is non-blank but not one of {sorted(CANONICAL_EVENT_TYPES)}.",
    detection="Membership check against the canonical event-type list.",
    severity=Severity.WARNING,
    action=Action.QUARANTINE,
    rationale="Unknown event types would be silently ignored by downstream consumers.",
)
def invalid_event_type(data: RawData) -> pd.Series:
    et = data.port_events["event_type"].fillna("")
    return ~_blank(et) & ~et.str.strip().isin(CANONICAL_EVENT_TYPES)


@register(
    "port_events.sentinel_vessel_id",
    "port_events",
    description=f"vessel_id is a placeholder / test value ({', '.join(sorted(SENTINEL_VESSEL_IDS))}) "
    "that never appears in shipments.",
    detection="Set difference of port_events.vessel_id minus shipments.vessel_id returned exactly these two ids.",
    severity=Severity.WARNING,
    action=Action.QUARANTINE,
    rationale="Events for test vessels would pollute port-level delay metrics.",
)
def sentinel_vessel_id(data: RawData) -> pd.Series:
    return data.port_events["vessel_id"].str.strip().isin(SENTINEL_VESSEL_IDS)


@register(
    "port_events.unknown_port_code",
    "port_events",
    description="port_code is not present in ports.csv.",
    detection="Anti-join against ports.port_code.",
    severity=Severity.WARNING,
    action=Action.QUARANTINE,
    rationale="Cannot be attributed to a known port.",
)
def events_unknown_port(data: RawData) -> pd.Series:
    return ~data.port_events["port_code"].str.strip().isin(_known_ports(data))


@register(
    "port_events.unparseable_timestamp",
    "port_events",
    description="event_timestamp is blank or not YYYY-MM-DD HH:MM:SS.",
    detection="Strict datetime parse.",
    severity=Severity.CRITICAL,
    action=Action.QUARANTINE,
    rationale="Events without a valid time cannot be ordered in the stream.",
)
def events_unparseable_timestamp(data: RawData) -> pd.Series:
    return _ts(data.port_events["event_timestamp"]).isna()


@register(
    "port_events.out_of_order",
    "port_events",
    description="event_timestamp is earlier than the previous row's (the feed is expected in time order).",
    detection="Row-wise diff of parsed event_timestamp < 0.",
    severity=Severity.INFO,
    action=Action.NONE,
    rationale="The curated layer sorts by timestamp anyway; late arrivals matter once this is a real stream "
    "(watermarks / allowed lateness). Currently passing.",
)
def events_out_of_order(data: RawData) -> pd.Series:
    return _ts(data.port_events["event_timestamp"]).diff().dt.total_seconds() < 0


@register(
    "port_events.delay_out_of_range",
    "port_events",
    description=f"delay_minutes is non-numeric, negative, or above {MAX_PLAUSIBLE_EVENT_DELAY_MINUTES}.",
    detection="Numeric parse + range check (observed range 0-300).",
    severity=Severity.WARNING,
    action=Action.FLAG,
    rationale="Negative or multi-day per-event delays are almost certainly unit errors.",
)
def event_delay_out_of_range(data: RawData) -> pd.Series:
    d = _num(data.port_events["delay_minutes"])
    return d.isna() | (d < 0) | (d > MAX_PLAUSIBLE_EVENT_DELAY_MINUTES)


@register(
    "port_events.delayed_event_without_delay",
    "port_events",
    description="event_type is DELAYED but delay_minutes is 0.",
    detection="Cross-tab of event_type vs delay_minutes > 0.",
    severity=Severity.INFO,
    action=Action.FLAG,
    rationale="Contradictory but harmless for current consumers; flagged so producers can be asked about it.",
)
def delayed_event_without_delay(data: RawData) -> pd.Series:
    e = data.port_events
    return (e["event_type"].str.strip() == "DELAYED") & (_num(e["delay_minutes"]) == 0)


@register(
    "port_events.notes_contradict_delay",
    "port_events",
    description="notes says 'Normal operations' but delay_minutes > 0.",
    detection="Cross-tab of notes vs delay_minutes; notes appear uniformly distributed across event types.",
    severity=Severity.INFO,
    action=Action.NONE,
    rationale="Free-text notes look unrelated to the structured fields, so they are kept for display but must "
    "not be used as a feature or to explain delays.",
)
def notes_contradict_delay(data: RawData) -> pd.Series:
    e = data.port_events
    return (e["notes"].str.strip() == "Normal operations") & (_num(e["delay_minutes"]) > 0)


@register(
    "port_events.missing_notes",
    "port_events",
    description="notes is blank.",
    detection="Blank check on notes.",
    severity=Severity.INFO,
    action=Action.NONE,
    rationale="Free-text notes are optional; kept as NULL.",
)
def missing_notes(data: RawData) -> pd.Series:
    return _blank(data.port_events["notes"])
