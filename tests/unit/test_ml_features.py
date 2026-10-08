import math

import pandas as pd
import pytest

from ml.features import (
    BOOKING_TIME_INPUTS,
    FEATURE_COLUMNS,
    LEAKY_COLUMNS,
    PortInfo,
    build_features,
    build_target,
)
from ml.predictor import risk_band

PORTS = {"CNSHA": PortInfo("APAC", 0.85), "NLRTM": PortInfo("EUR", 0.80)}


def booking(**overrides):
    row = {
        "origin_port": "CNSHA",
        "destination_port": "NLRTM",
        "cargo_type": "Electronics",
        "container_count": 10,
        "weight_tons": 500.0,
        "booking_date": "2025-01-01 00:00:00",
        "planned_departure": "2025-01-08 12:00:00",
        "planned_arrival": "2025-02-07 12:00:00",
    }
    row.update(overrides)
    return row


def test_feature_frame_has_expected_columns_in_order():
    X = build_features(pd.DataFrame([booking()]), PORTS)
    assert list(X.columns) == list(FEATURE_COLUMNS)


def test_derived_values():
    X = build_features(pd.DataFrame([booking()]), PORTS).iloc[0]
    assert X["booking_lead_days"] == 7.5
    assert X["transit_days_planned"] == 30.0
    assert X["origin_region"] == "APAC"
    assert X["region_lane"] == "APAC->EUR"
    assert X["origin_congestion"] == 0.85
    assert X["departure_month"] == 1


def test_no_leakage_even_if_actuals_are_supplied():
    leaky = booking(
        actual_arrival="2025-02-20 00:00:00", actual_departure="2025-01-09", status="DELAYED", actual_delay_hours=300.0
    )
    X = build_features(pd.DataFrame([leaky]), PORTS)
    assert not set(X.columns) & LEAKY_COLUMNS
    # and the actuals have no influence on any feature value
    assert X.equals(build_features(pd.DataFrame([booking()]), PORTS))


def test_leaky_columns_are_not_booking_inputs():
    assert not set(BOOKING_TIME_INPUTS) & LEAKY_COLUMNS


def test_unknown_port_and_missing_numerics_are_tolerated():
    X = build_features(pd.DataFrame([booking(origin_port="ZZZZZ", weight_tons=None, cargo_type=None)]), PORTS).iloc[0]
    assert X["origin_region"] == "UNKNOWN"
    assert math.isnan(X["origin_congestion"])
    assert math.isnan(X["weight_tons"])
    assert X["cargo_type"] == "Unknown"


def test_missing_input_column_raises():
    with pytest.raises(ValueError, match="Missing booking-time inputs"):
        build_features(pd.DataFrame([{"origin_port": "CNSHA"}]), PORTS)


def test_target_boundary_is_strictly_greater_than_24h():
    assert build_target(pd.Series([-3.0, 24.0, 24.5, 100.0])).tolist() == [0, 0, 1, 1]


@pytest.mark.parametrize(("p", "band"), [(0.9, "HIGH"), (0.5, "HIGH"), (0.4, "MEDIUM"), (0.35, "MEDIUM"), (0.2, "LOW")])
def test_risk_band(p, band):
    assert risk_band(p, threshold=0.5) == band
