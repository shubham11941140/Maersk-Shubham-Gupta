"""Structured interaction log: every LLM call, tool call, guardrail event and turn summary as one
JSON line in ``logs/assistant/assistant-YYYYMMDD.jsonl``.

Per LLM call we record the full prompt sent (messages + system-prompt hash; the system prompt itself is
logged once per session), the tools offered, the raw response blocks, token usage, cost and latency.
Per tool call: arguments, the exact result returned to the model, latency and errors.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class InteractionLogger:
    def __init__(self, log_dir: Path, session_id: str) -> None:
        self.session_id = session_id
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._dir / f"assistant-{datetime.now(UTC):%Y%m%d}.jsonl"

    def log(self, record_type: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "type": record_type,
            "session_id": self.session_id,
            **fields,
        }
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return record
