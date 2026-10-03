"""Grok (xAI) reasoner — live HTTP. Not used in pytest."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import Any

from tot.config import TOT_K
from tot.providers.http_util import chat_completion

DEFAULT_GROK_BASE_URL = "https://api.x.ai/v1"
DEFAULT_GROK_MODEL = "grok-4.6"


class GrokReasoner:
    """ToT expand/score + GI→NL proof via xAI Chat Completions."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = (
            api_key or os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "GROK_API_KEY is not set. Export GROK_API_KEY (or XAI_API_KEY) "
                "to use GrokReasoner. pytest must use MockReasoner."
            )
        self.model = model or os.environ.get("GROK_MODEL") or DEFAULT_GROK_MODEL
        self.score_model = (
            os.environ.get("GROK_SCORE_MODEL")
            or os.environ.get("CHEAP_MODEL")
            or "grok-4.3"
        )
        self.base_url = (
            base_url or os.environ.get("GROK_BASE_URL") or DEFAULT_GROK_BASE_URL
        )
        self.reasoning_effort = os.environ.get("GROK_REASONING_EFFORT", "low")

    def expand(self, proof_state: str, last_claim: str, k: int = TOT_K) -> list[str]:
        prompt = (
            "You are expanding a tree-of-thought for a math proof.\n"
            f"Current informal proof state:\n{proof_state}\n\n"
            f"Last claim:\n{last_claim}\n\n"
            f"Propose exactly {k} distinct next claims. "
            f"Return a JSON array of {k} strings and nothing else."
        )
        content = chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": "You propose next mathematical claims."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            extra_body={"reasoning_effort": self.reasoning_effort},
        )
        claims = _parse_claim_list(content, k)
        if len(claims) < k:
            claims.extend(
                f"Continue from: {last_claim}" for _ in range(k - len(claims))
            )
        return claims[:k]

    def score(self, proof_state: str, claim: str) -> float:
        prompt = (
            "Rate how feasible this next claim is for completing a correct proof.\n"
            f"Proof state:\n{proof_state}\n\nClaim:\n{claim}\n\n"
            "Reply with a single float in [0, 1] only."
        )
        content = chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.score_model,
            messages=[
                {"role": "system", "content": "You score proof-step feasibility."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            extra_body={
                "max_tokens": 32,
                "reasoning_effort": "low",
            },
        )
        match = re.search(r"-?\d+(?:\.\d+)?", content)
        if not match:
            return 0.5
        value = float(match.group(0))
        return max(0.0, min(1.0, value))

    def prove_nl(self, gi_payload: Mapping[str, Any] | str) -> str:
        if isinstance(gi_payload, str):
            payload_text = gi_payload
        else:
            payload_text = json.dumps(gi_payload, ensure_ascii=False, indent=2)
        prompt = (
            "Write a clear natural-language math proof using ONLY the claims in this "
            "support-core GI JSON (ids, statements, dependencies). "
            "Do not invent claims outside GI.\n\n"
            f"{payload_text}"
        )
        return chat_completion(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You write human-readable proofs from a claim DAG core.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            extra_body={"reasoning_effort": self.reasoning_effort},
        )


def _parse_claim_list(content: str, k: int) -> list[str]:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start >= 0 and end > start:
        try:
            data = json.loads(stripped[start : end + 1])
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()]
        except json.JSONDecodeError:
            pass
    lines = [
        re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
        for line in stripped.splitlines()
        if line.strip()
    ]
    return [line for line in lines if line][:k]
