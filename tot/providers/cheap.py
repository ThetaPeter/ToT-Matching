"""Cheap extractor — live HTTP (NL → ClaimDAG JSON). Not used in pytest."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

from tot.providers.http_util import chat_completion, extract_json_object

DEFAULT_CHEAP_BASE_URL = "https://api.x.ai/v1"
DEFAULT_CHEAP_MODEL = "grok-4.3"


_EXTRACT_SYSTEM = """You convert informal math text into a ProofFlow-style claim DAG JSON.
Return ONLY JSON with shape:
{"problem_id": str, "nodes": [{"id": str, "natural_language": str, "statement": str, "dependencies": [str]}]}
Use ids like tc_1 (condition), l1/l2 (lemmas), ts_1 (theorem solution). Dependencies list prerequisite claim ids.
No markdown fences."""


class CheapExtractor:
    """Requires ``CHEAP_API_KEY`` or ``EXTRACTOR_API_KEY`` (falls back to xAI Grok keys)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("CHEAP_API_KEY")
            or os.environ.get("EXTRACTOR_API_KEY")
            or os.environ.get("GROK_API_KEY")
            or os.environ.get("XAI_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "CHEAP_API_KEY is not set. Export CHEAP_API_KEY (or EXTRACTOR_API_KEY), "
                "or reuse GROK_API_KEY / XAI_API_KEY. pytest must use MockExtractor."
            )
        self.model = model or os.environ.get("CHEAP_MODEL") or DEFAULT_CHEAP_MODEL
        self.base_url = (
            base_url or os.environ.get("CHEAP_BASE_URL") or DEFAULT_CHEAP_BASE_URL
        )

    def extract(
        self,
        theorem: str,
        proof: str,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        ctx = dict(context or {})
        problem_id = str(ctx.get("problem_id") or "problem")
        existing = ctx.get("existing")
        if existing is not None:
            existing_text = (
                existing if isinstance(existing, str) else json.dumps(existing)
            )
            user = (
                f"problem_id={problem_id}\n"
                f"Incremental expansion. Existing DAG JSON:\n{existing_text}\n\n"
                f"Current proof state:\n{theorem}\n\n"
                f"New claim text:\n{proof}\n\n"
                "Return a JSON ClaimDAG fragment for the NEW nodes only "
                "(same schema), including any updated ts_* sink."
            )
        else:
            user = (
                f"problem_id={problem_id}\n"
                f"Theorem:\n{theorem}\n\n"
                f"Informal proof:\n{proof}\n\n"
                "Return the full ClaimDAG JSON."
            )
        content = chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        data = extract_json_object(content)
        if isinstance(data, dict):
            data.setdefault("problem_id", problem_id)
        return json.dumps(data)

    def repair(
        self,
        dag_json: str,
        issues: Sequence[Any],
        context: Mapping[str, Any] | None = None,
    ) -> str:
        ctx = dict(context or {})
        problem_id = str(ctx.get("problem_id") or "problem")
        issue_text = "\n".join(str(item) for item in issues) or "(no issues listed)"
        user = (
            f"problem_id={problem_id}\n"
            f"Repair this ClaimDAG JSON.\nIssues:\n{issue_text}\n\n"
            f"Current JSON:\n{dag_json}\n\n"
            "Return corrected ClaimDAG JSON only."
        )
        content = chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        data = extract_json_object(content)
        if isinstance(data, dict):
            data.setdefault("problem_id", problem_id)
        return json.dumps(data)
