"""Optional: the brief's three questions against the REAL Claude API.

Skipped unless ANTHROPIC_API_KEY is set (it costs money and is non-deterministic), so CI stays
hermetic. Run with:  ANTHROPIC_API_KEY=... pytest -m live_llm
"""

from __future__ import annotations

import os

import pytest

from assistant.config import AssistantSettings
from assistant.factory import build_assistant

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


@pytest.mark.parametrize(
    ("question", "expected_tool"),
    [
        ("Which routes have had the highest average delay this quarter?", "rank_routes"),
        ("What is the on-time rate for shipments from Shanghai to Rotterdam?", "get_route_stats"),
        ("What is the delay risk for shipment SHP-00421?", "predict_delay"),
    ],
)
def test_example_questions(base_url, tmp_path, question, expected_tool):
    bot = build_assistant(AssistantSettings(api_url=base_url, log_dir=tmp_path))
    ans = bot.ask(question)
    assert ans.status == "answered", ans.text
    assert expected_tool in {s.tool for s in ans.sources}
    assert "[S" in ans.text


def test_off_domain_question_is_refused(base_url, tmp_path):
    bot = build_assistant(AssistantSettings(api_url=base_url, log_dir=tmp_path))
    ans = bot.ask("What's a good recipe for banana bread?")
    assert ans.status == "refused"
    assert ans.tool_calls == 0
