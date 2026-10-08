from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "artifacts" / "model"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark tests by folder so ``-m unit`` / ``-m integration`` just work."""
    for item in items:
        path = Path(str(item.fspath))
        if "integration" in path.parts:
            item.add_marker(pytest.mark.integration)
        elif "unit" in path.parts:
            item.add_marker(pytest.mark.unit)
