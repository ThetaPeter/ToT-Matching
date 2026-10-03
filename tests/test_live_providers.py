"""HTTP provider and Lean compile tests (mocked transport; no network)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tot.lean.compile import CompileResult, compile_lean_code
from tot.providers.cheap import CheapExtractor
from tot.providers.deepseek_prover import DeepSeekProver
from tot.providers.grok import GrokReasoner


def test_grok_expand_parses_json_array(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    def fake_chat(**kwargs: Any) -> str:
        del kwargs
        return '["claim A", "claim B", "claim C"]'

    monkeypatch.setattr("tot.providers.grok.chat_completion", fake_chat)
    reasoner = GrokReasoner()
    claims = reasoner.expand("state", "last", k=3)
    assert claims == ["claim A", "claim B", "claim C"]


def test_grok_score_uses_grok_43(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    seen: dict[str, Any] = {}

    def fake_chat(**kwargs: Any) -> str:
        seen["model"] = kwargs.get("model")
        seen["extra_body"] = kwargs.get("extra_body")
        return "0.75"

    monkeypatch.setattr("tot.providers.grok.chat_completion", fake_chat)
    reasoner = GrokReasoner()
    assert reasoner.score("state", "claim") == 0.75
    assert seen["model"] == "grok-4.3"
    assert seen["extra_body"]["max_tokens"] == 32


def test_cheap_extract_returns_claimdag_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHEAP_API_KEY", "test-key")
    payload = {
        "problem_id": "p",
        "nodes": [
            {
                "id": "tc_1",
                "natural_language": "goal",
                "statement": "goal",
                "dependencies": [],
            },
            {
                "id": "l1",
                "natural_language": "step",
                "statement": "step",
                "dependencies": ["tc_1"],
            },
            {
                "id": "ts_1",
                "natural_language": "done",
                "statement": "done",
                "dependencies": ["l1"],
            },
        ],
    }

    def fake_chat(**kwargs: Any) -> str:
        del kwargs
        return json.dumps(payload)

    monkeypatch.setattr("tot.providers.cheap.chat_completion", fake_chat)
    extractor = CheapExtractor()
    raw = extractor.extract("thm", "proof", context={"problem_id": "p"})
    data = json.loads(raw)
    assert data["problem_id"] == "p"
    assert {node["id"] for node in data["nodes"]} == {"tc_1", "l1", "ts_1"}


def test_deepseek_prover_uses_injectable_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_chat(**kwargs: Any) -> str:
        del kwargs
        return "theorem gi_goal : True := by\n  trivial"

    def fake_compile(code: str, formal_statement: str | None = None) -> CompileResult:
        del formal_statement
        assert "trivial" in code
        return CompileResult(ok=True, errors=[])

    monkeypatch.setattr("tot.providers.deepseek_prover.chat_completion", fake_chat)
    prover = DeepSeekProver(compile_fn=fake_compile)
    result = prover.prove(
        {
            "problem_id": "mathd_algebra_10",
            "nodes": [
                {
                    "id": "tc_1",
                    "statement": "P",
                    "dependencies": [],
                }
            ],
        }
    )
    assert result["ok"] is True
    assert result["errors"] == []
    assert "trivial" in result["code"]


def test_compile_lean_code_reports_missing_toolchain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tot.lean.compile.shutil.which", lambda _name: None)
    result = compile_lean_code("theorem t : True := by trivial")
    assert result.ok is False
    assert result.errors
    assert "not found" in result.errors[0].lower() or "Lean 4" in result.errors[0]
