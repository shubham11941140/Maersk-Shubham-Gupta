import json

import pytest

from assistant.agent import Assistant
from assistant.config import AssistantSettings
from assistant.guardrails import FALLBACK_ANSWER
from assistant.observability import InteractionLogger
from assistant.prompts import build_system_prompt
from assistant.tools import ToolRegistry
from tests.fakes import FakeSupplyChainApi, ScriptedLLM, last_tool_result, text, tool_use


@pytest.fixture
def settings(tmp_path):
    return AssistantSettings(log_dir=tmp_path / "logs", max_tool_calls_per_question=3, anthropic_api_key="x")


def make(settings, script, api=None):
    llm = ScriptedLLM(script)
    api = api or FakeSupplyChainApi()
    system = build_system_prompt(api.ports(), "2024-Q2", "2025-Q4 (partial)")
    bot = Assistant(llm, ToolRegistry(api), system, settings, InteractionLogger(settings.log_dir, "sess"))
    return bot, llm, api


def records(settings):
    [path] = list(settings.log_dir.glob("*.jsonl"))
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_tool_call_then_grounded_answer(settings):
    bot, llm, api = make(
        settings,
        [
            [tool_use("get_route_stats", {"origin": "CNSHA", "destination": "NLRTM"})],
            lambda m: [text(f"The on-time rate is 75% [{last_tool_result(m)['source_id']}].")],
        ],
    )
    ans = bot.ask("On-time rate Shanghai to Rotterdam?")
    assert ans.status == "answered"
    assert ans.text == "The on-time rate is 75% [S1]."
    assert (ans.tool_calls, ans.llm_calls) == (1, 2)
    assert [s.source_id for s in ans.sources] == ["S1"]
    assert ans.cost_usd == pytest.approx(2 * (1000 * 0.10 + 100 * 0.50) / 1e6)
    # second request carried the tool_result back to the model
    tool_msg = llm.requests[1]["messages"][-1]
    assert tool_msg["role"] == "user"
    assert tool_msg["content"][0]["type"] == "tool_result"


def test_every_llm_and_tool_call_is_logged(settings):
    bot, _, _ = make(
        settings,
        [
            [tool_use("predict_delay", {"shipment_id": "SHP-00421"})],
            [text("Risk is MEDIUM with probability 0.42 [S1]; low-confidence model.")],
        ],
    )
    bot.ask("Delay risk for SHP-00421?")
    recs = records(settings)
    kinds = [r["type"] for r in recs]
    assert kinds == ["session_start", "llm_call", "tool_call", "llm_call", "turn"]
    start, call1, tool, _, turn = recs
    assert "Port codes" in start["system_prompt"]
    assert call1["request"]["messages"][0]["content"] == "Delay risk for SHP-00421?"
    assert call1["request"]["tools_offered"] == ["query_shipments", "get_route_stats", "rank_routes", "predict_delay"]
    assert call1["response"]["content"][0]["name"] == "predict_delay"
    assert call1["usage"]["input_tokens"] == 1000
    assert tool["output"]["delay_probability"] == 0.42
    assert turn["final_answer"].startswith("Risk is MEDIUM")
    assert turn["cited_sources"] == ["S1"]
    assert all(r["session_id"] == "sess" for r in recs)


def test_ungrounded_answer_is_repaired(settings):
    bot, llm, _ = make(
        settings,
        [
            [tool_use("get_route_stats", {"origin": "CNSHA", "destination": "NLRTM"})],
            [text("The on-time rate is 75%.")],  # no citation
            [text("The on-time rate is 75% [S1].")],
        ],
    )
    ans = bot.ask("On-time rate?")
    assert ans.status == "answered"
    assert "grounding_failed" in ans.guardrail_events
    assert "grounding check" in llm.requests[2]["messages"][-1]["content"]


def test_repeated_grounding_failure_falls_back(settings):
    bot, _, _ = make(settings, [[text("It is 99%.")], [text("Still 99% [S5].")]])
    ans = bot.ask("Make something up")
    assert ans.status == "fallback"
    assert ans.text == FALLBACK_ANSWER
    assert ans.guardrail_events.count("grounding_failed") == 2


def test_out_of_scope_refusal(settings):
    bot, _, api = make(settings, [[text("OUT_OF_SCOPE: I can only help with the shipment data.")]])
    ans = bot.ask("Write me a poem about cats")
    assert ans.status == "refused"
    assert ans.text == "I can only help with the shipment data."
    assert api.calls == []
    assert "out_of_scope" in ans.guardrail_events


def test_tool_budget_is_enforced_in_code(settings):
    greedy = [[tool_use("get_route_stats", {"origin": "CNSHA", "destination": "NLRTM"}, f"tu_{i}")] for i in range(4)]
    bot, llm, api = make(settings, [*greedy, [text("Done: 75% [S1].")]])
    ans = bot.ask("loop forever")
    assert ans.tool_calls == 3  # budget
    assert len(api.calls) == 3
    assert "tool_budget_exceeded" in ans.guardrail_events
    assert llm.requests[-1]["allow_tools"] is False  # tool_choice none once the budget is spent
    assert ans.status == "answered"


def test_llm_failure_rolls_back_turn(settings):
    bot, _, _ = make(settings, [RuntimeError("upstream 529"), [text("Hello again, no numbers here.")]])
    ans = bot.ask("first")
    assert ans.status == "error"
    assert bot._messages == []  # history is not left half-written
    assert bot.ask("second").status == "answered"


def test_rejected_input_never_calls_the_llm(settings):
    bot, llm, _ = make(settings, [])
    assert bot.ask("   ").status == "rejected"
    assert bot.ask("x" * 5000).status == "rejected"
    assert llm.requests == []


def test_sources_persist_across_turns(settings):
    bot, _, _ = make(
        settings,
        [
            [tool_use("get_route_stats", {"origin": "CNSHA", "destination": "NLRTM"})],
            [text("75% [S1].")],
            [text("As I said, 75% [S1].")],  # follow-up may cite an earlier turn's source
        ],
    )
    bot.ask("rate?")
    assert bot.ask("repeat that").status == "answered"


def test_system_prompt_contents():
    prompt = build_system_prompt(
        [{"port_code": "CNSHA", "port_name": "Shanghai", "country": "China", "region": "APAC"}],
        "2024-Q2",
        "2025-Q4 (partial)",
    )
    assert "CNSHA: Shanghai" in prompt
    assert "OUT_OF_SCOPE:" in prompt
    assert "[S1]" in prompt
    assert "2025-Q4 (partial)" in prompt
