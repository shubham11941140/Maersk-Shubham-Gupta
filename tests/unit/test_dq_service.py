import json
import os

import pytest

from app.errors import DependencyUnavailableError
from app.services.data_quality_service import DataQualityReportService

REPORT = {
    "generated_at": "2025-01-01T00:00:00+00:00",
    "summary": {"checks_run": 3},
    "checks": [
        {"check_name": "a", "dataset": "shipments", "severity": "critical", "status": "fail"},
        {"check_name": "b", "dataset": "shipments", "severity": "info", "status": "pass"},
        {"check_name": "c", "dataset": "port_events", "severity": "warning", "status": "fail"},
    ],
}


@pytest.fixture
def report_path(tmp_path):
    path = tmp_path / "dq.json"
    path.write_text(json.dumps(REPORT))
    return path


def test_missing_report_is_service_unavailable(tmp_path):
    with pytest.raises(DependencyUnavailableError):
        DataQualityReportService(tmp_path / "nope.json").report()


def test_corrupt_report_is_service_unavailable(tmp_path):
    path = tmp_path / "dq.json"
    path.write_text("{not json")
    with pytest.raises(DependencyUnavailableError, match="corrupt"):
        DataQualityReportService(path).report()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, ["a", "b", "c"]),
        ({"severity": "critical"}, ["a"]),
        ({"dataset": "port_events"}, ["c"]),
        ({"only_failed": True}, ["a", "c"]),
    ],
)
def test_filters(report_path, kwargs, expected):
    result = DataQualityReportService(report_path).report(**kwargs)
    assert [c["check_name"] for c in result["checks"]] == expected
    assert result["summary"] == REPORT["summary"]  # summary is never filtered


def test_cache_refreshes_when_file_changes(report_path):
    svc = DataQualityReportService(report_path)
    assert svc.summary()["checks_run"] == 3
    report_path.write_text(json.dumps({**REPORT, "summary": {"checks_run": 99}}))
    stat = report_path.stat()
    os.utime(report_path, (stat.st_atime, stat.st_mtime + 5))
    assert svc.summary()["checks_run"] == 99
