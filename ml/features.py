"""Booking-time feature engineering.

The SAME function is used for training and for serving, which removes the most
common source of train/serve skew. Only information known when the booking is
made is used — never actual_departure / actual_arrival / status.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from shared.reference import ON_TIME_THRESHOLD_HOURS

# Columns the caller must supply (all known at booking time).
BOOKING_TIME_INPUTS: tuple[str, ...] = (
    "origin_port",
    "destination_port",
    "cargo_type",
    "container_count",
    "weight_tons",
    "booking_date",
    "planned_departure",
    "planned_arrival",
)

# Columns that must never reach the model (leakage guard, asserted in tests).
LEAKY_COLUMNS: frozenset[str] = frozenset(
    {"actual_departure", "actual_arrival", "actual_delay_hours", "on_time_flag", "status", "transit_days_actual"}
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "origin_port",
    "destination_port",
    "cargo_type",
    "origin_region",
    "destination_region",
    "region_lane",
)
NUMERIC_FEATURES: tuple[str, ...] = (
    "container_count",
    "weight_tons",
    "booking_lead_days",
    "transit_days_planned",
    "origin_congestion",
    "destination_congestion",
    "departure_month",
    "departure_dayofweek",
    "booking_dayofweek",
)
FEATURE_COLUMNS: tuple[str, ...] = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Why each feature is in the model. Every one is known when the booking is made.
FEATURE_DOCS: dict[str, str] = {
    "origin_port": "Port-specific operations (congestion, labour, weather exposure).",
    "destination_port": "Same, at the discharge end — where an arrival delay is actually measured.",
    "cargo_type": "Inspection-heavy cargo (pharma, chemicals, food) can be held at customs.",
    "origin_region": "Coarser version of origin_port; helps ports with few shipments.",
    "destination_region": "Coarser version of destination_port.",
    "region_lane": "Trade lane (e.g. APAC->EUR): route-level effects without 600 sparse route dummies.",
    "container_count": "Larger consignments need more handling slots.",
    "weight_tons": "Heavy cargo can be rolled to a later vessel when capacity is tight.",
    "booking_lead_days": "Late bookings get worse slots; long lead times give planners slack.",
    "transit_days_planned": "Longer voyages have more opportunity to accumulate delay.",
    "origin_congestion": "Reference congestion score of the origin port (ports.csv).",
    "destination_congestion": "Reference congestion score of the destination port.",
    "departure_month": "Seasonality (peak season, monsoon, winter storms).",
    "departure_dayofweek": "Weekend departures can face reduced port staffing.",
    "booking_dayofweek": "Proxy for booking process / channel.",
}

# Deliberately excluded, with the reason (leakage or no booking-time availability).
EXCLUDED_FEATURES: dict[str, str] = {
    "actual_departure / actual_arrival / actual_delay_hours / transit_days_actual": "Leakage: only known after "
    "the fact — the target is derived from them.",
    "status / on_time_flag": "Leakage: derived from the outcome.",
    "shipment_id": "An identifier, not a property of the shipment.",
    "customer_id / vessel_id (raw)": "Hundreds of sparse ids → memorisation. Their point-in-time history was "
    "tested instead (ml/history_features.py) and gave no lift.",
    "port_events (same window)": "Events during the voyage are not known at booking. Events before booking "
    "were tested as point-in-time features — no lift.",
}


@dataclass(frozen=True)
class PortInfo:
    region: str
    congestion: float


PortLookup = Mapping[str, PortInfo]


def build_features(df: pd.DataFrame, ports: PortLookup) -> pd.DataFrame:
    """Turn booking-time inputs into the model's feature frame."""
    missing = [c for c in BOOKING_TIME_INPUTS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing booking-time inputs: {missing}")

    booking = pd.to_datetime(df["booking_date"])
    p_dep = pd.to_datetime(df["planned_departure"])
    p_arr = pd.to_datetime(df["planned_arrival"])

    def region(code: str) -> str:
        info = ports.get(code)
        return info.region if info else "UNKNOWN"

    def congestion(code: str) -> float:
        info = ports.get(code)
        return info.congestion if info else np.nan

    out = pd.DataFrame(index=df.index)
    out["origin_port"] = df["origin_port"].astype(str)
    out["destination_port"] = df["destination_port"].astype(str)
    out["cargo_type"] = df["cargo_type"].fillna("Unknown").astype(str)
    out["origin_region"] = out["origin_port"].map(region)
    out["destination_region"] = out["destination_port"].map(region)
    out["region_lane"] = out["origin_region"] + "->" + out["destination_region"]

    out["container_count"] = pd.to_numeric(df["container_count"], errors="coerce").astype(float)
    out["weight_tons"] = pd.to_numeric(df["weight_tons"], errors="coerce").astype(float)
    out["booking_lead_days"] = (p_dep - booking).dt.total_seconds() / 86400.0
    out["transit_days_planned"] = (p_arr - p_dep).dt.total_seconds() / 86400.0
    out["origin_congestion"] = out["origin_port"].map(congestion).astype(float)
    out["destination_congestion"] = out["destination_port"].map(congestion).astype(float)
    out["departure_month"] = p_dep.dt.month.astype(float)
    out["departure_dayofweek"] = p_dep.dt.dayofweek.astype(float)
    out["booking_dayofweek"] = booking.dt.dayofweek.astype(float)
    return out[list(FEATURE_COLUMNS)]


def build_target(delay_hours: pd.Series, threshold_hours: float = ON_TIME_THRESHOLD_HOURS) -> pd.Series:
    """1 when the shipment arrived more than ``threshold_hours`` late."""
    return (delay_hours > threshold_hours).astype(int)
