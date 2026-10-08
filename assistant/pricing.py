"""Token prices for cost-per-query reporting (USD per million tokens, standard API rates).

Source (checked Oct 2026): OpenAI's model pages, e.g. https://developers.openai.com/api/docs/models/gpt-5.6-luna
and https://developers.openai.com/api/docs/pricing. Cache writes on GPT-5.6 bill at 1.25× input.
Prices change — override with SCI_ASSISTANT_PRICE_INPUT / _CACHED_INPUT / _OUTPUT if needed.
Models not listed (e.g. a local Ollama model) report cost as n/a.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

CACHE_WRITE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    cached_input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, Price] = {
    "gpt-5.6-luna": Price(0.20, 0.02, 1.20),
    "gpt-5.6-terra": Price(2.00, 0.20, 12.00),
}


def price_for(model: str) -> Price | None:
    env_in = os.getenv("SCI_ASSISTANT_PRICE_INPUT")
    env_out = os.getenv("SCI_ASSISTANT_PRICE_OUTPUT")
    if env_in and env_out:
        env_cached = os.getenv("SCI_ASSISTANT_PRICE_CACHED_INPUT", env_in)
        return Price(float(env_in), float(env_cached), float(env_out))
    return PRICES.get(model)


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """``input_tokens`` is the total prompt size; cached / cache-write tokens are subsets of it."""
    price = price_for(model)
    if price is None:
        return None
    uncached = max(input_tokens - cached_input_tokens - cache_write_tokens, 0)
    total = (
        uncached * price.input_per_mtok
        + cached_input_tokens * price.cached_input_per_mtok
        + cache_write_tokens * price.input_per_mtok * CACHE_WRITE_MULTIPLIER
        + output_tokens * price.output_per_mtok
    )
    return round(total / 1_000_000, 6)
