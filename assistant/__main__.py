"""CLI.

python -m assistant                                   # interactive session (type 'exit' to quit)
python -m assistant ask "Which routes had the highest average delay this quarter?"
python -m assistant demo                              # the three example questions from the brief
python -m assistant --model gpt-5.6-terra ask "..."
"""

from __future__ import annotations

import argparse
import sys

from assistant.agent import Answer
from assistant.api_client import ApiError
from assistant.config import AssistantSettings
from assistant.factory import AssistantConfigError, build_assistant

DEMO_QUESTIONS = (
    "Which routes have had the highest average delay this quarter?",
    "What is the on-time rate for shipments from Shanghai to Rotterdam?",
    "What is the delay risk for shipment SHP-00421?",
)


def render(answer: Answer, verbose: bool = True) -> str:
    out = [answer.text]
    if verbose:
        if answer.sources:
            out.append("")
            out.append("Sources:")
            for s in answer.sources:
                args = ", ".join(f"{k}={v}" for k, v in s.input.items() if v not in (None, {}, []))
                out.append(f"  [{s.source_id}] {s.tool}({args})")
        cost = f"${answer.cost_usd:.4f}" if answer.cost_usd is not None else "n/a"
        out.append(
            f"\n— {answer.status} · {answer.llm_calls} LLM call(s) · {answer.tool_calls} tool call(s) · "
            f"{answer.input_tokens}+{answer.output_tokens} tokens · ≈{cost} · {answer.latency_ms / 1000:.1f}s"
            + (f" · guardrails: {', '.join(answer.guardrail_events)}" if answer.guardrail_events else "")
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m assistant", description="Supply Chain Intelligence assistant")
    parser.add_argument("--model", help="Override SCI_ASSISTANT_MODEL (default gpt-5.6-luna)")
    parser.add_argument("--api-url", help="Override SCI_ASSISTANT_API_URL")
    parser.add_argument("--quiet", action="store_true", help="Answer only, no sources / cost footer")
    sub = parser.add_subparsers(dest="command")
    ask = sub.add_parser("ask", help="Ask one question")
    ask.add_argument("question", nargs="+")
    sub.add_parser("demo", help="Run the brief's three example questions")
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in {"model": args.model, "api_url": args.api_url}.items() if v}
    settings = AssistantSettings(**overrides)
    try:
        bot = build_assistant(settings)
    except AssistantConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ApiError as exc:
        print(f"error: cannot reach the data API at {settings.api_url} ({exc}). Is it running?", file=sys.stderr)
        return 2

    print(f"[model {settings.model} · data API {settings.api_url} · log {bot._log.path}]\n", file=sys.stderr)
    if args.command == "ask":
        print(render(bot.ask(" ".join(args.question)), not args.quiet))
        return 0
    if args.command == "demo":
        for q in DEMO_QUESTIONS:
            print(f"> {q}\n{render(bot.ask(q), not args.quiet)}\n")
        return 0

    print("Ask about shipments, routes, on-time performance or delay risk. 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in {"exit", "quit", ":q"}:
            return 0
        if q:
            print(render(bot.ask(q), not args.quiet))


if __name__ == "__main__":
    sys.exit(main())
