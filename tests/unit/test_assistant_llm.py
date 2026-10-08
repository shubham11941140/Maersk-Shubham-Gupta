"""OpenAI adapter: canonical ⇄ Chat Completions translation, request shape, usage and pricing."""

import json
from types import SimpleNamespace

import pytest

from assistant import llm as llm_module
from assistant.config import AssistantSettings
from assistant.factory import AssistantConfigError, build_assistant
from assistant.llm import from_openai_response, to_openai_messages, to_openai_tools
from assistant.pricing import cost_usd
from assistant.tools import TOOLS
from tests.fakes import FakeSupplyChainApi

HISTORY = [
    {"role": "user", "content": "Risk for SHP-00421?"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "call_1", "name": "predict_delay", "input": {"shipment_id": "SHP-00421"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": '{"ok": true}', "is_error": False},
        ],
    },
    {"role": "user", "content": "Your previous answer failed a check."},
]


def test_canonical_history_maps_to_chat_completions_messages():
    msgs = to_openai_messages("SYSTEM", HISTORY)
    assert msgs[0] == {"role": "system", "content": "SYSTEM"}
    assert msgs[1] == {"role": "user", "content": "Risk for SHP-00421?"}
    assistant = msgs[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Checking."
    assert assistant["tool_calls"][0]["function"] == {
        "name": "predict_delay",
        "arguments": json.dumps({"shipment_id": "SHP-00421"}),
    }
    assert msgs[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'}
    assert msgs[4] == {"role": "user", "content": "Your previous answer failed a check."}


def test_assistant_message_without_text_has_null_content():
    msgs = to_openai_messages(
        "S", [{"role": "assistant", "content": [{"type": "tool_use", "id": "c", "name": "rank_routes", "input": {}}]}]
    )
    assert msgs[1]["content"] is None
    assert len(msgs[1]["tool_calls"]) == 1


def test_tool_schemas_map_to_function_definitions():
    tools = to_openai_tools([t.schema() for t in TOOLS])
    assert {t["function"]["name"] for t in tools} == {t.name for t in TOOLS}
    assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in tools)


def fake_completion(content=None, tool_calls=(), finish="tool_calls"):
    return SimpleNamespace(
        id="chatcmpl_1",
        model="gpt-5.6-luna",
        choices=[
            SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content, tool_calls=list(tool_calls)))
        ],
        usage=SimpleNamespace(
            prompt_tokens=2400,
            completion_tokens=180,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2048, cache_write_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=64),
        ),
    )


def call(cid, name, arguments):
    return SimpleNamespace(id=cid, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def test_response_maps_back_to_canonical_blocks():
    resp = from_openai_response(
        fake_completion(
            content="Let me check.",
            tool_calls=[
                call("call_1", "get_route_stats", '{"origin": "CNSHA", "destination": "NLRTM"}'),
                call("call_2", "predict_delay", '{"shipment_id": "SHP-00421"}'),
            ],  # parallel tool calls
        )
    )
    assert resp.text == "Let me check."
    assert [t["name"] for t in resp.tool_uses] == ["get_route_stats", "predict_delay"]
    assert resp.tool_uses[0]["input"] == {"origin": "CNSHA", "destination": "NLRTM"}
    assert (resp.usage.input_tokens, resp.usage.cached_input_tokens, resp.usage.reasoning_tokens) == (2400, 2048, 64)
    assert resp.stop_reason == "tool_calls"


def test_malformed_tool_arguments_become_a_validation_error_not_a_crash():
    resp = from_openai_response(fake_completion(tool_calls=[call("c", "predict_delay", "{not json")]))
    assert resp.tool_uses[0]["input"] == {"__invalid_arguments__": "{not json"}
    from assistant.tools import ToolRegistry

    result = ToolRegistry(FakeSupplyChainApi()).execute("predict_delay", resp.tool_uses[0]["input"])
    assert result.payload["code"] == "INVALID_ARGUMENTS"


def test_adapter_sends_a_well_formed_request(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return fake_completion(content="Done [S1].", finish="stop")

    def fake_client(**kw):
        captured["client_kwargs"] = kw
        return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    monkeypatch.setitem(__import__("sys").modules, "openai", SimpleNamespace(OpenAI=fake_client))
    client = llm_module.OpenAILLM(
        api_key="k", model="gpt-5.6-luna", reasoning_effort="low", base_url="http://localhost:11434/v1"
    )
    resp = client.create("SYSTEM", HISTORY[:1], [t.schema() for t in TOOLS], allow_tools=False)

    assert resp.text == "Done [S1]."
    assert captured["tool_choice"] == "none"
    assert captured["reasoning_effort"] == "low"
    assert captured["max_completion_tokens"] == 2048
    assert "temperature" not in captured  # GPT-5.x reasoning models reject non-default temperature
    assert captured["messages"][0]["role"] == "system"
    assert captured["client_kwargs"]["base_url"] == "http://localhost:11434/v1"
    assert resp.request["messages"] == captured["messages"]  # exact payload is exposed for logging


def test_reasoning_effort_can_be_disabled_for_non_reasoning_models(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return fake_completion(content="ok", finish="stop")

    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        SimpleNamespace(OpenAI=lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))),
    )
    llm_module.OpenAILLM(api_key="k", model="llama3.1", reasoning_effort="").create("S", HISTORY[:1], [])
    assert "reasoning_effort" not in captured


def test_pricing_includes_cached_input():
    # 1M prompt tokens of which 800k cached, 1M output on gpt-5.6-luna
    assert cost_usd("gpt-5.6-luna", 1_000_000, 1_000_000, cached_input_tokens=800_000) == pytest.approx(
        0.2 * 0.20 + 0.8 * 0.02 + 1.20
    )
    assert cost_usd("gpt-5.6-terra", 10_000, 1_000) == pytest.approx(0.032)
    assert cost_usd("llama3.1", 1, 1) is None


def test_price_override(monkeypatch):
    monkeypatch.setenv("SCI_ASSISTANT_PRICE_INPUT", "1")
    monkeypatch.setenv("SCI_ASSISTANT_PRICE_OUTPUT", "2")
    assert cost_usd("anything", 1_000_000, 1_000_000) == 3.0


def test_missing_api_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AssistantConfigError, match="OPENAI_API_KEY"):
        build_assistant(AssistantSettings(log_dir=tmp_path, _env_file=None), api=FakeSupplyChainApi())
