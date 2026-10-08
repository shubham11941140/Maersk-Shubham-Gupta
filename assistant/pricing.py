"""Token prices for cost-per-query reporting (USD per million tokens, standard API rates).

Sources (checked Oct 2026): https://www.anthropic.com/claude/haiku (Haiku 5.5, prompts ≤ 100K tokens)
and https://platform.claude.com/docs/en/about-claude/pricing. Prices change — override with
SCI_ASSISTANT_PRICE_INPUT / SCI_ASSISTANT_PRICE_OUTPUT if needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50),
    "claude-sonnet-5-5": Price(2.00, 10.00),
    "claude-opus-5-5": Price(4.00, 20.00),
}


def price_for(model: str) -> Price | None:
    override_in, override_out = os.getenv("SCI_ASSISTANT_PRICE_INPUT"), os.getenv("SCI_ASSISTANT_PRICE_OUTPUT")
    if override_in and override_out:
        return Price(float(override_in), float(override_out))
    return PRICES.get(model)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = price_for(model)
    if price is None:
        return None
    return round((input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000, 6)
