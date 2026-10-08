from datetime import datetime

import pytest

from shared.domain import delay_hours, is_on_time, route_key, transit_days
from shared.reference import canonical_cargo_type, canonical_status


class TestDelayHours:
    def test_late_arrival_is_positive(self):
        assert delay_hours(datetime(2025, 1, 1, 0), datetime(2025, 1, 2, 6)) == 30.0

    def test_early_arrival_is_negative(self):
        assert delay_hours(datetime(2025, 1, 1, 12), datetime(2025, 1, 1, 4)) == -8.0

    @pytest.mark.parametrize(
        ("planned", "actual"), [(None, datetime(2025, 1, 1)), (datetime(2025, 1, 1), None), (None, None)]
    )
    def test_null_safe(self, planned, actual):
        assert delay_hours(planned, actual) is None


class TestOnTime:
    @pytest.mark.parametrize(
        ("delay", "expected"), [(-5.0, True), (0.0, True), (24.0, True), (24.01, False), (100.0, False)]
    )
    def test_threshold_is_inclusive_at_24h(self, delay, expected):
        assert is_on_time(delay) is expected

    def test_null_delay_is_unknown_not_false(self):
        assert is_on_time(None) is None

    def test_custom_threshold(self):
        assert is_on_time(10.0, threshold_hours=6) is False


def test_route_key_format():
    assert route_key("CNSHA", "NLRTM") == "CNSHA → NLRTM"


def test_transit_days():
    assert transit_days(datetime(2025, 1, 1), datetime(2025, 1, 11, 12)) == 10.5
    assert transit_days(None, datetime(2025, 1, 1)) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FURNITURE", "Furniture"),
        (" Chemicals", "Chemicals"),
        ("machinery ", "Machinery"),
        ("Automotive parts", "Automotive Parts"),
        ("food  &  beverage", "Food & Beverage"),
        ("", None),
        (None, None),
        ("Spaceships", None),
    ],
)
def test_canonical_cargo_type(raw, expected):
    assert canonical_cargo_type(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("delivered", "DELIVERED"),
        ("Complete", "DELIVERED"),
        ("COMPLETED", "DELIVERED"),
        ("DELAYED", "DELAYED"),
        ("N/A", None),
        ("", None),
        (None, None),
    ],
)
def test_canonical_status(raw, expected):
    assert canonical_status(raw) == expected
