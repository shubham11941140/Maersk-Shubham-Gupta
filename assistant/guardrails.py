"""Deterministic guardrails around the LLM.

1. Input guard      — empty / oversized questions are rejected before any LLM call.
2. Scope guard      — the model is instructed to answer only supply-chain questions about this
                      dataset and to prefix refusals with ``OUT_OF_SCOPE:``; refusals are detected,
                      logged and shown without the marker.
3. Budget guard     — caps on tool calls per question and per session, and on LLM calls per question
                      (stops runaway loops and bounds cost). Enforced in code, not in the prompt.
4. Grounding guard  — any answer containing numbers must cite at least one tool result as [S#],
                      and every cited id must exist. Violations trigger one repair turn, then a
                      safe fallback. This stops the model presenting invented figures as facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OUT_OF_SCOPE_MARKER = "OUT_OF_SCOPE:"
_CITATION = re.compile(r"\[(S\d+)\]")
_IGNORED_TOKENS = re.compile(
    r"SHP-\d+|CUST-\d+|VSL-\d+|\[S\d+\]|\b\d{4}-Q[1-4]\b|\b[A-Z]{5}\b"
)  # ids, citations, quarter labels and port codes are not "claims"
_NUMBER = re.compile(r"\d")


class InputRejected(ValueError):
    pass


def check_question(question: str, max_chars: int) -> str:
    q = (question or "").strip()
    if not q:
        raise InputRejected("Please ask a question.")
    if len(q) > max_chars:
        raise InputRejected(f"Question is too long ({len(q)} characters, max {max_chars}).")
    return q


def is_refusal(answer: str) -> bool:
    return answer.lstrip().upper().startswith(OUT_OF_SCOPE_MARKER)


def strip_refusal_marker(answer: str) -> str:
    text = answer.lstrip()
    return text[len(OUT_OF_SCOPE_MARKER) :].strip() if is_refusal(text) else answer


@dataclass(frozen=True)
class GroundingVerdict:
    ok: bool
    cited: frozenset[str]
    reason: str | None = None


def check_grounding(answer: str, known_sources: set[str]) -> GroundingVerdict:
    if is_refusal(answer):
        return GroundingVerdict(True, frozenset())
    cited = frozenset(_CITATION.findall(answer))
    unknown = cited - known_sources
    if unknown:
        return GroundingVerdict(False, cited, f"cites sources that do not exist: {sorted(unknown)}")
    has_numbers = bool(_NUMBER.search(_IGNORED_TOKENS.sub(" ", answer)))
    if has_numbers and not cited:
        return GroundingVerdict(False, cited, "contains figures but cites no tool result")
    return GroundingVerdict(True, cited)


REPAIR_INSTRUCTION = (
    "Your previous answer failed an automatic grounding check ({reason}). Rewrite it so that every figure "
    "comes from a tool result and is followed by its source id in square brackets, e.g. [S1]. Use only "
    "source ids that appear in tool results. If you do not have the data, call a tool or say you don't know."
)

FALLBACK_ANSWER = (
    "I couldn't produce an answer I can back up with the data tools, so I'm not going to guess. "
    "Try rephrasing, or ask about a specific shipment, route or quarter."
)
