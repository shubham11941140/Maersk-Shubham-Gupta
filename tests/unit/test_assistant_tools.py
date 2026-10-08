import json

from assistant.api_client import ApiError
from assistant.tools import TOOLS, ToolRegistry
from tests.fakes import FakeSupplyChainApi


def registry(api=None, **kw):
    return ToolRegistry(api or FakeSupplyChainApi(), **kw)


def test_the_required_tools_exist_with_self_contained_schemas():
    names = {t.name for t in TOOLS}
    assert {"query_shipments", "get_route_stats", "predict_delay"} <= names
    for schema in registry().schemas():
        dumped = json.dumps(schema)
        assert "$ref" not in dumped
        assert schema["input_schema"]["type"] == "object"
        assert schema["description"]


def test_query_by_shipment_id_uses_detail_endpoint():
    api = FakeSupplyChainApi()
    r = registry(api).execute("query_shipments", {"filters": {"shipment_id": "SHP-00421"}})
    assert r.ok
    assert r.source_id == "S1"
    assert api.calls == [("get_shipment", "SHP-00421")]
    assert r.payload["items"][0]["shipment_id"] == "SHP-00421"


def test_query_with_filters_maps_to_list_endpoint_and_normalises():
    api = FakeSupplyChainApi()
    r = registry(api).execute(
        "query_shipments", {"filters": {"origin": "cnsha", "status": "DELAYED", "date_from": "2025-01-01"}, "limit": 3}
    )
    name, params = api.calls[0]
    assert name == "list_shipments"
    assert params["origin"] == "CNSHA"
    assert params["page_size"] == 3
    assert params["date_from"] == "2025-01-01"
    assert r.payload["total_matching"] == 42
    assert set(r.payload["items"][0]) <= {
        "shipment_id", "route_key", "status", "cargo_type", "container_count", "booking_date",
        "planned_departure", "planned_arrival", "actual_arrival", "actual_delay_hours", "on_time_flag",
    }  # fmt: skip


def test_route_stats_and_predict_delay():
    api = FakeSupplyChainApi()
    reg = registry(api)
    assert reg.execute("get_route_stats", {"origin": "CNSHA", "destination": "nlrtm"}).payload["on_time_rate"] == 0.75
    p = reg.execute("predict_delay", {"shipment_id": "SHP-00421"})
    assert p.source_id == "S2"
    assert "low-confidence" in p.payload["model_quality_caveat"]


def test_rank_routes_compacts_periods():
    r = registry().execute("rank_routes", {})
    assert r.payload["available_periods"] == ["2025-Q2", "2025-Q3", "2025-Q4 (partial)"]


def test_invalid_arguments_never_reach_the_api():
    api = FakeSupplyChainApi()
    reg = registry(api)
    bad = [
        ("get_route_stats", {"origin": "SHANGHAI", "destination": "NLRTM"}),
        ("predict_delay", {"shipment_id": "421"}),
        ("query_shipments", {"filters": {"status": "LOST"}}),
        ("query_shipments", {"limit": 500}),
        ("query_shipments", {"sql": "DROP TABLE x"}),  # unknown argument -> forbidden
    ]
    for name, args in bad:
        r = reg.execute(name, args)
        assert not r.ok
        assert r.payload["code"] == "INVALID_ARGUMENTS"
        assert r.source_id is None
    assert api.calls == []


def test_unknown_tool_and_api_errors_are_returned_not_raised():
    assert registry().execute("drop_database", {}).payload["code"] == "UNKNOWN_TOOL"
    api = FakeSupplyChainApi(fail_with=ApiError(404, "SHIPMENT_NOT_FOUND", "nope"))
    r = registry(api).execute("predict_delay", {"shipment_id": "SHP-99999"})
    assert not r.ok
    assert r.payload["code"] == "SHIPMENT_NOT_FOUND"
    assert json.loads(r.to_model_content())["ok"] is False


def test_large_results_are_truncated():
    r = registry(max_result_chars=500).execute("query_shipments", {"limit": 3})
    assert r.truncated
    assert len(r.payload["items"]) < 3
    assert r.payload["returned"] == len(r.payload["items"])
