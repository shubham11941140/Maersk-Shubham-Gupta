"""Assistant settings — environment variables with prefix ``SCI_ASSISTANT_``.

OpenAI credentials use the SDK's standard names: ``OPENAI_API_KEY`` and (optionally)
``OPENAI_BASE_URL`` for Azure OpenAI, OpenRouter, or a local Ollama / vLLM server.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AssistantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCI_ASSISTANT_", env_file=".env", extra="ignore")

    # LLM
    model: str = "gpt-5.6-luna"
    # GPT-5.6 models reason before answering; "low" keeps tool routing fast and cheap.
    # Set to "" for models without a reasoning_effort parameter (e.g. many local models).
    reasoning_effort: str = "low"
    max_tokens: int = 2048  # includes reasoning tokens
    request_timeout_s: float = 60.0
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    # Data product
    api_url: str = "http://localhost:8000"
    api_timeout_s: float = 10.0

    # Guardrails
    max_question_chars: int = 2000
    max_tool_calls_per_question: int = 6
    max_tool_calls_per_session: int = 40
    max_llm_calls_per_question: int = 8
    max_grounding_repairs: int = 1
    max_tool_result_chars: int = 6000

    # Observability
    log_dir: Path = Path("logs/assistant")
