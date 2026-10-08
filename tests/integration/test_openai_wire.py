"""The real ``openai`` SDK against a local OpenAI-compatible stub server.

Proves the adapter's requests serialise through the actual SDK and that real SDK response
objects parse back correctly — including a full assistant turn with a tool call against the
live data API. No network, no key, no cost.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from assistant.api_client import SupplyChainApi
from assistant.config import AssistantSettings
from assistant.factory import build_assistant
from assistant.llm import OpenAILLM


class _StubOpenAI(BaseHTTPRequestHandler):
    requests: list[dict] = []  # noqa: RUF012 - shared capture list for the test

    def log_message(self, *args):  # silence the default stderr logging
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        last = body["messages"][-1]
        if last["role"] == "tool":  # second call: answer citing the tool result
            source = json.loads(last["content"])["source_id"]
            rate = json.loads(last["content"])["data"]["on_time_rate"]
            message = {"role": "assistant", "content": f"The on-time rate is {rate:.0%} [{source}]."}
            finish = "stop"
        else:  # first call: ask for a tool
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_route_stats",
                            "arguments": '{"origin": "CNSHA", "destination": "NLRTM"}',
                        },
                    }
                ],
            }
            finish = "tool_calls"
        payload = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": 2500,
                "completion_tokens": 120,
                "total_tokens": 2620,
                "prompt_tokens_details": {"cached_tokens": 2048},
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def stub_openai():
    _StubOpenAI.requests = []
    server = HTTPServer(("127.0.0.1", 0), _StubOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", _StubOpenAI.requests
    server.shutdown()


def test_full_turn_through_the_real_sdk(stub_openai, base_url, client, tmp_path):
    stub_url, captured = stub_openai
    llm = OpenAILLM(api_key="sk-test", model="gpt-5.6-luna", reasoning_effort="low", base_url=stub_url)
    settings = AssistantSettings(api_url=base_url, log_dir=tmp_path, openai_api_key="sk-test")
    bot = build_assistant(settings, llm=llm, api=SupplyChainApi(base_url))

    ans = bot.ask("What is the on-time rate for shipments from Shanghai to Rotterdam?")

    expected = client.get("/routes/CNSHA/NLRTM/stats").json()["on_time_rate"]
    assert ans.status == "answered"
    assert ans.text == f"The on-time rate is {expected:.0%} [S1]."
    assert ans.cached_input_tokens == 2 * 2048
    assert ans.cost_usd is not None

    first, second = captured[0], captured[1]
    assert first["path"] == "/v1/chat/completions"
    assert first["auth"] == "Bearer sk-test"
    assert first["body"]["tool_choice"] == "auto"
    assert first["body"]["reasoning_effort"] == "low"
    assert {t["function"]["name"] for t in first["body"]["tools"]} >= {
        "query_shipments",
        "get_route_stats",
        "predict_delay",
    }
    # second request replays the assistant tool call and the tool result in OpenAI's format
    roles = [m["role"] for m in second["body"]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second["body"]["messages"][2]["tool_calls"][0]["id"] == "call_abc"
    assert second["body"]["messages"][3]["tool_call_id"] == "call_abc"
