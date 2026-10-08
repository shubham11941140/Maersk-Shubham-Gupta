"""LLM provider abstraction. The agent depends on the ``LLMClient`` Protocol only, so the
provider can be swapped (or faked in tests) without touching the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]  # normalised blocks: {"type": "text", "text"} | {"type": "tool_use", ...}
    stop_reason: str | None
    usage: Usage = field(default_factory=Usage)
    model: str | None = None
    response_id: str | None = None

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


def _normalise_block(block: Any) -> dict[str, Any] | None:
    kind = getattr(block, "type", None)
    if kind == "text":
        return {"type": "text", "text": block.text}
    if kind == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input or {})}
    return None  # e.g. thinking blocks — not used here


class AnthropicLLM:
    """Claude via the official SDK (Messages API with native tool use)."""

    def __init__(
        self, api_key: str, model: str, max_tokens: int = 1024, temperature: float = 0.0, timeout_s: float = 60.0
    ) -> None:
        import anthropic  # imported lazily so tests / the API image don't need a key

        self.model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=2)

    def create(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]], allow_tools: bool = True
    ) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            # the system prompt + tool schemas are identical on every call → cache them
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"} if allow_tools else {"type": "none"},
        )
        u = resp.usage
        return LLMResponse(
            content=[b for b in (_normalise_block(x) for x in resp.content) if b],
            stop_reason=resp.stop_reason,
            usage=Usage(
                input_tokens=getattr(u, "input_tokens", 0) or 0,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
            ),
            model=resp.model,
            response_id=resp.id,
        )
