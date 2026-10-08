"""Assistant settings — environment variables with prefix ``SCI_ASSISTANT_`` (API key: ``ANTHROPIC_API_KEY``)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AssistantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCI_ASSISTANT_", env_file=".env", extra="ignore")

    # LLM
    model: str = "claude-haiku-5-5"
    max_tokens: int = 1024
    temperature: float = 0.0  # deterministic-ish answers over data
    request_timeout_s: float = 60.0
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

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
