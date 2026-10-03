"""DeepSeek-Prover via OpenRouter + optional Lean 4 typecheck. Not used in pytest."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any

from tot.lean.compile import CompileFn, CompileResult, compile_lean_code
from tot.lean.statements import formal_statement_for
from tot.providers.http_util import chat_completion

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PROVER_MODEL = "deepseek/deepseek-prover-v2"


class DeepSeekProver:
    """GI JSON → Lean code, then real Lean 4 compile when available."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        compile_fn: CompileFn | None = None,
        skip_compile: bool | None = None,
        formal_statement: str | None = None,
        problem_id: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("DEEPSEEK_PROVER_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Export DEEPSEEK_API_KEY "
                "(or DEEPSEEK_PROVER_API_KEY / OPENROUTER_API_KEY) to use DeepSeekProver. "
                "pytest must use MockProver."
            )
        self.model = (
            model or os.environ.get("DEEPSEEK_PROVER_MODEL") or DEFAULT_PROVER_MODEL
        )
        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_PROVER_BASE_URL")
            or os.environ.get("OPENROUTER_BASE_URL")
            or DEFAULT_OPENROUTER_BASE_URL
        )
        self.compile_fn = compile_fn or compile_lean_code
        if skip_compile is None:
            skip_compile = os.environ.get("TOT_SKIP_LEAN_COMPILE", "").lower() in {
                "1",
                "true",
                "yes",
            }
        self.skip_compile = bool(skip_compile)
        self.formal_statement = formal_statement
        self.problem_id = problem_id

    def prove(self, gi_payload: Mapping[str, Any] | str) -> Mapping[str, Any]:
        payload = (
            json.loads(gi_payload) if isinstance(gi_payload, str) else dict(gi_payload)
        )
        problem_id = str(
            self.problem_id
            or payload.get("problem_id")
            or payload.get("id")
            or "gi_goal"
        )
        statement = self.formal_statement or formal_statement_for(problem_id)
        code = self._generate_lean(payload, statement=statement, problem_id=problem_id)
        if self.skip_compile:
            return {"ok": False, "code": code, "errors": ["lean compile skipped"]}
        result = self._compile(code, statement)
        return {
            "ok": bool(result.ok),
            "code": code,
            "errors": list(result.errors),
        }

    def _generate_lean(
        self,
        payload: Mapping[str, Any],
        *,
        statement: str | None,
        problem_id: str,
    ) -> str:
        gi_text = json.dumps(payload, ensure_ascii=False, indent=2)
        formal = statement or (
            f"theorem {problem_id} : True := by\n  sorry"
        )
        prompt = (
            "You are DeepSeek-Prover. Write a Lean 4 proof for the formal statement "
            "using ONLY the informal support-core claims in GI as guidance.\n"
            "Return Lean 4 code only (no markdown).\n\n"
            f"Formal statement:\n{formal}\n\n"
            f"GI JSON:\n{gi_text}\n"
        )
        content = chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You emit Lean 4 theorem proofs. No markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            extra_headers={
                "HTTP-Referer": "https://github.com/tot-local",
                "X-Title": "ToT DeepSeek-Prover",
            },
        )
        return _strip_fences(content)

    def _compile(self, code: str, statement: str | None) -> CompileResult:
        if self.compile_fn is compile_lean_code:
            return compile_lean_code(code, formal_statement=statement)
        return self.compile_fn(code, statement)


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    # Drop a leading "lean" language tag line if present without fences
    stripped = re.sub(r"^lean\n", "", stripped, count=1, flags=re.IGNORECASE)
    return stripped.strip()
