from types import SimpleNamespace

import pytest

from assistant import llm as llm_module
from assistant.config import AssistantSettings
from assistant.factory import AssistantConfigError, build_assistant
from assistant.pricing import cost_usd
from tests.fakes import FakeSupplyChainApi


class FakeMessages:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            id="msg_1",
            model=kwargs["model"],
            stop_reason="tool_use",
            usage=SimpleNamespace(
                input_tokens=1200, output_tokens=80, cache_read_input_tokens=1000, cache_creation_input_tokens=0
            ),
            content=[
                SimpleNamespace(type="text", text="Let me check."),
                SimpleNamespace(type="tool_use", id="tu_1", name="predict_delay", input={"shipment_id": "SHP-00421"}),
                SimpleNamespace(type="thinking", thinking="..."),
            ],
        )


def test_anthropic_adapter_normalises_response_and_sends_guarded_request(monkeypatch):
    messages = FakeMessages()
    fake_sdk = SimpleNamespace(Anthropic=lambda **kw: SimpleNamespace(messages=messages, kw=kw))
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_sdk)

    client = llm_module.AnthropicLLM(api_key="k", model="claude-haiku-5-5")
    resp = client.create("SYSTEM", [{"role": "user", "content": "hi"}], [{"name": "t"}], allow_tools=False)

    assert resp.text == "Let me check."
    assert resp.tool_uses == [
        {"type": "tool_use", "id": "tu_1", "name": "predict_delay", "input": {"shipment_id": "SHP-00421"}}
    ]
    assert len(resp.content) == 2  # unknown block types dropped
    assert resp.usage.cache_read_input_tokens == 1000
    sent = messages.kwargs
    assert sent["tool_choice"] == {"type": "none"}
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["temperature"] == 0.0


def test_pricing():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("claude-sonnet-5-5", 10_000, 1_000) == pytest.approx(0.03)
    assert cost_usd("some-unknown-model", 1, 1) is None


def test_price_override(monkeypatch):
    monkeypatch.setenv("SCI_ASSISTANT_PRICE_INPUT", "1")
    monkeypatch.setenv("SCI_ASSISTANT_PRICE_OUTPUT", "2")
    assert cost_usd("anything", 1_000_000, 1_000_000) == 3.0


def test_missing_api_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AssistantConfigError, match="ANTHROPIC_API_KEY"):
        build_assistant(AssistantSettings(log_dir=tmp_path, _env_file=None), api=FakeSupplyChainApi())
