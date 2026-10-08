"""Assistant end-to-end against the REAL running API, with a scripted LLM (no API key, deterministic).

The script decides *which* tools to call; everything after that — argument validation, HTTP calls
to the live service, result shaping, source ids, grounding checks and JSONL logging — is real.
"""

from __future__ import annotations

import json

import pytest

from assistant.api_client import SupplyChainApi
from assistant.config import AssistantSettings
from assistant.factory import build_assistant
from tests.fakes import ScriptedLLM, last_tool_result, text, tool_use


@pytest.fixture
def settings(base_url, tmp_path):
    return AssistantSettings(api_url=base_url, log_dir=tmp_path / "logs", anthropic_api_key="unused")


def bot_with(settings, script):
    llm = ScriptedLLM(script)
    return build_assistant(settings, llm=llm, api=SupplyChainApi(settings.api_url)), llm


def test_system_prompt_is_built_from_live_reference_data(settings):
    bot, _ = bot_with(settings, [])
    assert "CNSHA: Shanghai" in bot._system
    assert "NLRTM: Rotterdam" in bot._system
    assert "(partial)" in bot._system  # data coverage comes from the API


def test_on_time_rate_question(settings, client):
    expected = client.get("/routes/CNSHA/NLRTM/stats").json()["on_time_rate"]

    def answer(messages):
        r = last_tool_result(messages)
        return [
            text(f"The on-time rate from Shanghai to Rotterdam is {r['data']['on_time_rate']:.0%} [{r['source_id']}].")
        ]

    bot, _ = bot_with(settings, [[tool_use("get_route_stats", {"origin": "CNSHA", "destination": "NLRTM"})], answer])
    ans = bot.ask("What is the on-time rate for shipments from Shanghai to Rotterdam?")
    assert ans.status == "answered"
    assert f"{expected:.0%}" in ans.text
    assert ans.sources[0].payload["on_time_rate"] == expected


def test_delay_risk_question_calls_the_ml_endpoint(settings, client):
    expected = client.post("/predict-delay", json={"shipment_id": "SHP-00421"}).json()["delay_probability"]
    bot, _ = bot_with(
        settings,
        [
            [tool_use("predict_delay", {"shipment_id": "SHP-00421"})],
            lambda m: [text(f"Risk band {last_tool_result(m)['data']['risk_band']} [S1] — low-confidence model.")],
        ],
    )
    ans = bot.ask("What is the delay risk for shipment SHP-00421?")
    assert ans.status == "answered"
    assert ans.sources[0].payload["delay_probability"] == expected


def test_this_quarter_question_uses_latest_complete_quarter(settings, client):
    latest = client.get("/routes/rankings").json()

    def answer(messages):
        r = last_tool_result(messages)["data"]
        top = r["items"][0]
        return [
            text(
                f"In {r['period']}, the latest complete quarter in the data, {top['route_key']} had the "
                f"highest average delay at {top['avg_delay_hours']} h [S1]."
            )
        ]

    bot, _ = bot_with(settings, [[tool_use("rank_routes", {"period": "latest"})], answer])
    ans = bot.ask("Which routes have had the highest average delay this quarter?")
    assert ans.status == "answered"
    assert latest["period"] in ans.text
    assert latest["items"][0]["route_key"] in ans.text


def test_unknown_shipment_error_flows_back_to_the_model(settings):
    def answer(messages):
        r = last_tool_result(messages)
        assert r["ok"] is False
        assert r["error"]["code"] == "SHIPMENT_NOT_FOUND"
        return [text("I couldn't find shipment SHP-99999 in the data.")]

    bot, _ = bot_with(settings, [[tool_use("predict_delay", {"shipment_id": "SHP-99999"})], answer])
    assert bot.ask("Delay risk for SHP-99999?").status == "answered"


def test_log_file_captures_the_whole_exchange(settings):
    bot, _ = bot_with(
        settings,
        [
            [tool_use("query_shipments", {"filters": {"origin": "CNSHA", "status": "DELAYED"}, "limit": 2})],
            lambda m: [text(f"{last_tool_result(m)['data']['total_matching']} delayed shipments left Shanghai [S1].")],
        ],
    )
    bot.ask("How many delayed shipments left Shanghai?")
    [path] = list(settings.log_dir.glob("*.jsonl"))
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["type"] for r in recs] == ["session_start", "llm_call", "tool_call", "llm_call", "turn"]
    tool = recs[2]
    assert tool["ok"] is True
    assert tool["output"]["returned"] == 2
    assert recs[-1]["cited_sources"] == ["S1"]
