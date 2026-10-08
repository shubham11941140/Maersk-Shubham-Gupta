import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.dependencies import get_analytics_service
from app.errors import InvalidRequestError
from app.main import create_app
from app.services.analytics_service import AnalyticsService
from tests.fakes import FakeAnalyticsRepository


def test_latest_resolves_to_latest_complete_quarter_with_note():
    repo = FakeAnalyticsRepository()
    r = AnalyticsService(repo).route_rankings()
    assert r.period == "2025-Q3"
    assert r.latest_complete_period == "2025-Q3"
    assert "partial" in r.note
    assert repo.calls[0][1] == ("avg_delay_hours", True, "2025-Q3", 3, 10)
    assert r.items[0].rank == 1
    assert r.items[0].avg_delay_hours == 50.12


def test_all_and_explicit_partial_period():
    repo = FakeAnalyticsRepository()
    svc = AnalyticsService(repo)
    assert svc.route_rankings(period="all").period == "all"
    assert repo.calls[-1][1][2] is None
    partial = svc.route_rankings(period="2025-Q4")
    assert partial.period_is_complete is False
    assert "partially" in partial.note


def test_unknown_period_lists_available():
    with pytest.raises(InvalidRequestError) as exc:
        AnalyticsService(FakeAnalyticsRepository()).route_rankings(period="2019-Q1")
    assert "2025-Q3" in exc.value.details["available_periods"]


def test_no_complete_quarter():
    repo = FakeAnalyticsRepository(
        periods_rows=[{"period": "2025-Q4", "complete": False, "shipments": 1, "completed": 1}]
    )
    with pytest.raises(InvalidRequestError):
        AnalyticsService(repo).route_rankings()


@pytest.fixture
def client(tmp_path):
    app = create_app(
        Settings(
            db_path=tmp_path / "x.duckdb", model_dir=tmp_path, dq_report_path=tmp_path / "r.json", log_level="WARNING"
        )
    )
    app.dependency_overrides[get_analytics_service] = lambda: AnalyticsService(FakeAnalyticsRepository())
    with TestClient(app) as c:
        yield c


def test_http_contract(client):
    assert client.get("/ports").json()["items"][0]["port_code"] == "CNSHA"
    body = client.get("/routes/rankings", params={"metric": "on_time_rate", "order": "asc"}).json()
    assert (body["metric"], body["order"], body["period"]) == ("on_time_rate", "asc", "2025-Q3")
    for params in ({"metric": "vibes"}, {"period": "Q3"}, {"limit": 0}, {"min_completed": 1000}):
        assert client.get("/routes/rankings", params=params).status_code == 422
