"""Twelve-factor configuration: everything comes from environment variables (prefix ``SCI_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCI_", env_file=".env", extra="ignore")

    service_name: str = "supply-chain-intel-api"
    environment: str = "local"
    build_sha: str = "dev"
    log_level: str = "INFO"

    host: str = "0.0.0.0"  # noqa: S104 - bind all interfaces inside the container
    port: int = 8000

    db_path: Path = Path("data/warehouse/supply_chain.duckdb")
    model_dir: Path = Path("artifacts/model")
    dq_report_path: Path = Path("artifacts/dq_report.json")

    default_page_size: int = Field(default=20, ge=1)
    max_page_size: int = Field(default=100, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
