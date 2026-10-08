"""Composition root for the assistant (wires settings → API client → tools → LLM → agent)."""

from __future__ import annotations

import uuid

from assistant.agent import Assistant
from assistant.api_client import SupplyChainApi
from assistant.config import AssistantSettings
from assistant.llm import LLMClient, OpenAILLM
from assistant.observability import InteractionLogger
from assistant.prompts import build_system_prompt
from assistant.tools import ToolRegistry


class AssistantConfigError(RuntimeError):
    pass


def data_coverage(api: SupplyChainApi) -> tuple[str, str]:
    """First / last quarter in the data, from the rankings endpoint's period list."""
    periods = api.route_rankings({"period": "all", "limit": 1})["available_periods"]
    return periods[0]["period"], periods[-1]["period"] + ("" if periods[-1]["complete"] else " (partial)")


def build_assistant(
    settings: AssistantSettings, llm: LLMClient | None = None, api: SupplyChainApi | None = None
) -> Assistant:
    api = api or SupplyChainApi(settings.api_url, settings.api_timeout_s)
    if llm is None:
        if not settings.openai_api_key:
            raise AssistantConfigError("OPENAI_API_KEY is not set. Export it (or put it in .env) and retry.")
        llm = OpenAILLM(
            api_key=settings.openai_api_key,
            model=settings.model,
            max_tokens=settings.max_tokens,
            reasoning_effort=settings.reasoning_effort,
            base_url=settings.openai_base_url,
            timeout_s=settings.request_timeout_s,
        )
    data_from, data_to = data_coverage(api)
    system = build_system_prompt(api.ports(), data_from, data_to)
    tools = ToolRegistry(api, max_result_chars=settings.max_tool_result_chars)
    logger = InteractionLogger(settings.log_dir, session_id=uuid.uuid4().hex[:12])
    return Assistant(llm, tools, system, settings, logger)
