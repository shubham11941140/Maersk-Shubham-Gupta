import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.logging_config import JsonFormatter, request_id_ctx
from app.middleware import RequestContextMiddleware


def test_json_formatter_includes_extras_and_request_id():
    token = request_id_ctx.set("req-1")
    try:
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("world",), None)
        record.status_code = 200
        payload = json.loads(JsonFormatter("svc").format(record))
    finally:
        request_id_ctx.reset(token)
    assert payload["message"] == "hello world"
    assert payload["service"] == "svc"
    assert payload["request_id"] == "req-1"
    assert payload["status_code"] == 200
    assert payload["level"] == "INFO"


def test_middleware_logs_method_path_status_latency(caplog):
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    with caplog.at_level(logging.INFO, logger="app.access"), TestClient(app) as client:
        resp = client.get("/ping?x=1")

    record = next(r for r in caplog.records if r.name == "app.access")
    assert (record.method, record.path, record.status_code, record.query) == ("GET", "/ping", 200, "x=1")
    assert record.latency_ms >= 0
    assert resp.headers["x-request-id"]
