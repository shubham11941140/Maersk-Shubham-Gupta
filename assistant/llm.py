"""LLM provider abstraction. The agent depends on the ``LLMClient`` Protocol only, so the
provider can be swapped (or faked in tests) without touching the agent loop.

Canonical message format
------------------------
The agent keeps conversation history in a small provider-neutral shape:

* ``{"role": "user", "content": "question text"}``
* ``{"role": "assistant", "content": [{"type": "text", "text": ...},
                                      {"type": "tool_use", "id": ..., "name": ..., "input": {...}}]}``
* ``{"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": "<json>",
                                  "is_error": bool}]}``

Each adapter translates that to and from its provider's wire format. Only the adapter knows
about OpenAI; the agent, tools, guardrails and logs don't.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Usage:
    input_tokens: int = 0  # total prompt tokens, including cached ones
    output_tokens: int = 0  # completion tokens, including reasoning tokens
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]  # canonical blocks: {"type": "text", "text"} | {"type": "tool_use", ...}
    stop_reason: str | None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    response_id: str | None = None
    request: dict[str, Any] | None = None  # exact payload sent to the provider (for the interaction log)

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b["type"] == "text").strip()

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b["type"] == "tool_use"]


class LLMClient(Protocol):
    model: str

    def create(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        allow_tools: bool = True,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# OpenAI (Chat Completions with function calling)
# --------------------------------------------------------------------------- #
def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        }
        for t in tools
    ]


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical history → Chat Completions messages."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        if msg["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            tool_calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                }
                for b in content
                if b["type"] == "tool_use"
            ]
            item: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                item["tool_calls"] = tool_calls
            out.append(item)
            continue
        # user turn made of tool results → one "tool" message per result
        for b in content:
            if b["type"] == "tool_result":
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
            elif b["type"] == "text":
                out.append({"role": "user", "content": b["text"]})
    return out


def _parse_arguments(raw: str | None) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string. Malformed JSON is passed through under a reserved key,
    which the tool's ``extra="forbid"`` schema rejects — the model then sees INVALID_ARGUMENTS and retries."""
    try:
        parsed = json.loads(raw or "{}")
        return parsed if isinstance(parsed, dict) else {"__invalid_arguments__": raw}
    except json.JSONDecodeError:
        return {"__invalid_arguments__": raw}


def from_openai_response(resp: Any) -> LLMResponse:
    choice = resp.choices[0]
    message = choice.message
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    for call in message.tool_calls or []:
        if getattr(call, "type", "function") != "function":
            continue
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": _parse_arguments(call.function.arguments),
            }
        )
    u = resp.usage
    prompt_details = getattr(u, "prompt_tokens_details", None)
    completion_details = getattr(u, "completion_tokens_details", None)
    return LLMResponse(
        content=blocks,
        stop_reason=choice.finish_reason,
        usage=Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
            cached_input_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            cache_write_tokens=getattr(prompt_details, "cache_write_tokens", 0) or 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        ),
        model=resp.model,
        response_id=resp.id,
    )


class OpenAILLM:
    """OpenAI via the official SDK (Chat Completions + function calling).

    Chat Completions rather than the Responses API: the history stays fully client-side
    (every prompt is logged verbatim, no server-side state), and the same adapter works
    with any OpenAI-compatible endpoint — Azure OpenAI, OpenRouter, or a local model via
    Ollama / vLLM — by setting ``OPENAI_BASE_URL``.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        reasoning_effort: str | None = "low",
        base_url: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        import openai  # imported lazily so tests / the API image path don't need a key

        self.model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort or None
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout_s, max_retries=2)

    def create(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], allow_tools: bool = True
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
            "tools": to_openai_tools(tools),
            "tool_choice": "auto" if allow_tools else "none",
            # reasoning tokens count against this budget, so it is larger than the visible answer
            "max_completion_tokens": self._max_tokens,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        resp = from_openai_response(self._client.chat.completions.create(**kwargs))
        resp.request = kwargs
        return resp
