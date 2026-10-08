"""The agent loop: LLM ⇄ tools, with guardrails and full observability.

question ─► input guard ─► LLM ─┬─► tool_use? ─► budget guard ─► validated tool ─► result [S#] ─┐
                                │                                                              │
                                └─► final text ─► grounding guard ─┬─ ok ─► answer             │
                                          ▲                        └─ fail ─► repair (×1) ─────┤
                                          └────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from assistant import guardrails as g
from assistant.config import AssistantSettings
from assistant.llm import LLMClient, LLMResponse
from assistant.observability import InteractionLogger, sha256
from assistant.pricing import cost_usd
from assistant.tools import ToolRegistry, ToolResult


@dataclass
class Answer:
    text: str
    status: str  # answered | refused | rejected | fallback | error
    sources: list[ToolResult] = field(default_factory=list)
    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    latency_ms: float = 0.0
    guardrail_events: list[str] = field(default_factory=list)
    turn_id: str = ""


class Assistant:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
        settings: AssistantSettings,
        logger: InteractionLogger,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._system = system_prompt
        self._settings = settings
        self._log = logger
        self._messages: list[dict[str, Any]] = []
        self._sources: dict[str, ToolResult] = {}
        self._session_tool_calls = 0
        self._log.log(
            "session_start",
            model=llm.model,
            system_prompt=system_prompt,
            system_prompt_sha256=sha256(system_prompt),
            tools=tools.schemas(),
            limits={
                "max_tool_calls_per_question": settings.max_tool_calls_per_question,
                "max_tool_calls_per_session": settings.max_tool_calls_per_session,
                "max_llm_calls_per_question": settings.max_llm_calls_per_question,
            },
        )

    # ------------------------------------------------------------------ #
    def _guardrail(self, answer: Answer, kind: str, **detail: Any) -> None:
        answer.guardrail_events.append(kind)
        self._log.log("guardrail", turn_id=answer.turn_id, guardrail=kind, **detail)

    def _call_llm(self, answer: Answer, allow_tools: bool) -> LLMResponse:
        start = time.perf_counter()
        resp = self._llm.create(self._system, self._messages, self._tools.schemas(), allow_tools=allow_tools)
        latency = round((time.perf_counter() - start) * 1000, 1)
        answer.llm_calls += 1
        answer.input_tokens += resp.usage.input_tokens
        answer.output_tokens += resp.usage.output_tokens
        call_cost = cost_usd(self._llm.model, resp.usage.input_tokens, resp.usage.output_tokens)
        self._log.log(
            "llm_call",
            turn_id=answer.turn_id,
            call_index=answer.llm_calls,
            model=self._llm.model,
            request={
                "system_prompt_sha256": sha256(self._system),
                "messages": self._messages,
                "tools_offered": self._tools.names(),
                "tool_choice": "auto" if allow_tools else "none",
            },
            response={"content": resp.content, "stop_reason": resp.stop_reason, "id": resp.response_id},
            usage=resp.usage.__dict__,
            cost_usd=call_cost,
            latency_ms=latency,
        )
        return resp

    def _run_tools(self, answer: Answer, tool_uses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for tu in tool_uses:
            over_question = answer.tool_calls >= self._settings.max_tool_calls_per_question
            over_session = self._session_tool_calls >= self._settings.max_tool_calls_per_session
            if over_question or over_session:
                scope = "question" if over_question else "session"
                self._guardrail(answer, "tool_budget_exceeded", scope=scope, tool=tu["name"], input=tu["input"])
                content = json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "code": "TOOL_BUDGET_EXCEEDED",
                            "message": f"Tool-call budget for this {scope} is used up. "
                            "Answer with the results you already have.",
                        },
                    }
                )
                results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": content, "is_error": True})
                continue
            result = self._tools.execute(tu["name"], tu["input"])
            answer.tool_calls += 1
            self._session_tool_calls += 1
            if result.ok and result.source_id:
                self._sources[result.source_id] = result
                answer.sources.append(result)
            self._log.log(
                "tool_call",
                turn_id=answer.turn_id,
                tool_use_id=tu["id"],
                tool=result.tool,
                input=result.input,
                ok=result.ok,
                source_id=result.source_id,
                output=result.payload,
                truncated=result.truncated,
                latency_ms=result.latency_ms,
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": result.to_model_content(),
                    "is_error": not result.ok,
                }
            )
        return results

    # ------------------------------------------------------------------ #
    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        answer = Answer(text="", status="answered", turn_id=uuid.uuid4().hex[:12])
        try:
            q = g.check_question(question, self._settings.max_question_chars)
        except g.InputRejected as exc:
            answer.text, answer.status = str(exc), "rejected"
            self._guardrail(answer, "input_rejected", reason=str(exc))
            return self._finish(answer, question, started)

        history_len = len(self._messages)
        self._messages.append({"role": "user", "content": q})
        repairs = 0
        try:
            while True:
                if answer.llm_calls >= self._settings.max_llm_calls_per_question:
                    self._guardrail(answer, "llm_call_budget_exceeded", llm_calls=answer.llm_calls)
                    answer.text, answer.status = g.FALLBACK_ANSWER, "fallback"
                    break
                tools_left = (
                    answer.tool_calls < self._settings.max_tool_calls_per_question
                    and self._session_tool_calls < self._settings.max_tool_calls_per_session
                )
                resp = self._call_llm(answer, allow_tools=tools_left)
                self._messages.append({"role": "assistant", "content": resp.content or [{"type": "text", "text": ""}]})

                if resp.tool_uses:
                    self._messages.append({"role": "user", "content": self._run_tools(answer, resp.tool_uses)})
                    continue

                text = resp.text
                if g.is_refusal(text):
                    answer.text, answer.status = g.strip_refusal_marker(text), "refused"
                    self._guardrail(answer, "out_of_scope", question=q)
                    break
                verdict = g.check_grounding(text, set(self._sources))
                if verdict.ok:
                    answer.text = text
                    break
                self._guardrail(
                    answer, "grounding_failed", reason=verdict.reason, rejected_answer=text, attempt=repairs + 1
                )
                if repairs >= self._settings.max_grounding_repairs:
                    answer.text, answer.status = g.FALLBACK_ANSWER, "fallback"
                    break
                repairs += 1
                self._messages.append({"role": "user", "content": g.REPAIR_INSTRUCTION.format(reason=verdict.reason)})
        except Exception as exc:  # LLM / network failure: never crash the CLI, always log
            self._log.log("error", turn_id=answer.turn_id, error=f"{type(exc).__name__}: {exc}")
            answer.text, answer.status = f"Sorry — the assistant failed: {type(exc).__name__}: {exc}", "error"
            del self._messages[history_len:]  # roll back the half-finished turn so the session stays valid
        return self._finish(answer, question, started)

    def _finish(self, answer: Answer, question: str, started: float) -> Answer:
        answer.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        answer.cost_usd = cost_usd(self._llm.model, answer.input_tokens, answer.output_tokens)
        cited = sorted(g.check_grounding(answer.text, set(self._sources)).cited)
        self._log.log(
            "turn",
            turn_id=answer.turn_id,
            question=question,
            final_answer=answer.text,
            status=answer.status,
            cited_sources=cited,
            tools_called=[{"source_id": s.source_id, "tool": s.tool, "input": s.input} for s in answer.sources],
            tool_calls=answer.tool_calls,
            llm_calls=answer.llm_calls,
            usage={"input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens},
            cost_usd=answer.cost_usd,
            latency_ms=answer.latency_ms,
            guardrail_events=answer.guardrail_events,
        )
        return answer
