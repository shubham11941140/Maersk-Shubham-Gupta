import pytest

from assistant import guardrails as g


class TestInputGuard:
    def test_strips_and_accepts(self):
        assert g.check_question("  hi  ", 100) == "hi"

    @pytest.mark.parametrize("q", ["", "   ", None])
    def test_empty_rejected(self, q):
        with pytest.raises(g.InputRejected):
            g.check_question(q, 100)

    def test_too_long_rejected(self):
        with pytest.raises(g.InputRejected, match="too long"):
            g.check_question("x" * 101, 100)


class TestScopeGuard:
    def test_refusal_detection_and_stripping(self):
        assert g.is_refusal("OUT_OF_SCOPE: I only answer shipping questions.")
        assert g.is_refusal("  out_of_scope: nope")
        assert g.strip_refusal_marker("OUT_OF_SCOPE: I only answer shipping questions.") == (
            "I only answer shipping questions."
        )
        assert not g.is_refusal("The on-time rate is 75% [S1].")


class TestGroundingGuard:
    def test_cited_numbers_pass(self):
        v = g.check_grounding("On-time rate is 75% [S1] over 12 shipments [S2].", {"S1", "S2"})
        assert v.ok
        assert v.cited == {"S1", "S2"}

    def test_numbers_without_citation_fail(self):
        v = g.check_grounding("The on-time rate is 75%.", {"S1"})
        assert not v.ok
        assert "cites no tool result" in v.reason

    def test_unknown_citation_fails(self):
        v = g.check_grounding("It is 75% [S7].", {"S1"})
        assert not v.ok
        assert "S7" in v.reason

    @pytest.mark.parametrize(
        "answer",
        [
            "I can't find shipment SHP-99999.",  # ids are not claims
            "Which route do you mean, CNSHA → NLRTM?",  # port codes are not claims
            "The latest complete quarter is 2025-Q3.",  # quarter labels are not claims
            "OUT_OF_SCOPE: I can only help with the shipping data.",
        ],
    )
    def test_non_numeric_answers_need_no_citation(self, answer):
        assert g.check_grounding(answer, set()).ok
