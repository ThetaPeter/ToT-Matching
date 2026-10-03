"""Pytest isolation: hashed embedder fallback must not pick up a host MiniLM path."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_tot_minilm_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOT_MINILM_PATH", raising=False)
