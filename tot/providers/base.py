"""Provider protocols and pytest mocks (no network)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from tot.ir.dump import dumps, loads
from tot.ir.schema import ClaimDAG, ClaimNode


class Reasoner(Protocol):
    """Strong reasoner: ToT expand/score and GI → natural-language proof."""

    def expand(self, proof_state: str, last_claim: str, k: int) -> list[str]:
        """Return ``k`` candidate next-claim texts."""

    def score(self, proof_state: str, claim: str) -> float:
        """Feasibility prior in ``[0, 1]``. Phase 1 uses LLM/mock scores, not NR."""

    def prove_nl(self, gi_payload: Mapping[str, Any] | str) -> str:
        """Natural-language proof from a GI-only payload."""


class Extractor(Protocol):
    """Cheap extractor: informal NL → ClaimDAG JSON."""

    def extract(
        self,
        theorem: str,
        proof: str,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        """Return ClaimDAG JSON (string)."""

    def repair(
        self,
        dag_json: str,
        issues: Sequence[Any],
        context: Mapping[str, Any] | None = None,
    ) -> str:
        """Return repaired ClaimDAG JSON given validator issues."""


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Map node texts to a ``(n, d)`` matrix."""


class Prover(Protocol):
    """Lean prover: GI JSON → code + check result. Must never see the search tree."""

    def prove(self, gi_payload: Mapping[str, Any] | str) -> Mapping[str, Any]:
        """Return ``{ok, code, errors}``."""


def _payload_dict(gi_payload: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(gi_payload, str):
        return json.loads(gi_payload)
    return dict(gi_payload)


def hashed_token_vectors(texts: Sequence[str], dim: int = 32) -> np.ndarray:
    """Deterministic token-hash vectors (no Python ``hash()``, no network)."""
    mat = np.zeros((len(texts), dim), dtype=float)
    for i, text in enumerate(texts):
        for tok in (text or "").lower().split():
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:2], "little") % dim
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            mat[i, idx] += sign
        norm = float(np.linalg.norm(mat[i]))
        if norm > 0.0:
            mat[i] /= norm
        else:
            mat[i, 0] = 1.0
    return mat


class MockEmbedder:
    """MiniLM stand-in for tests: hashed token overlap, no downloads."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return hashed_token_vectors(texts, dim=self.dim)


class MockReasoner:
    def __init__(self, *, include_dead: bool = False) -> None:
        self.include_dead = include_dead
        self._step = 0
        self.expand_calls = 0
        self.prove_nl_calls = 0
        self.last_prove_payload: dict[str, Any] | None = None

    def expand(self, proof_state: str, last_claim: str, k: int) -> list[str]:
        self.expand_calls += 1
        base = (last_claim or proof_state or "claim").strip()
        claims: list[str] = []
        for j in range(max(0, k)):
            self._step += 1
            claims.append(
                f"Advance the argument (step {self._step}): continue from {base[:80]}"
            )
        if self.include_dead and claims:
            claims[-1] = f"dead-end unused case {self._step}: discard this alternative"
        return claims

    def score(self, proof_state: str, claim: str) -> float:
        del proof_state
        text = (claim or "").lower()
        if "dead" in text or "unused" in text:
            return 0.1
        return 0.9

    def prove_nl(self, gi_payload: Mapping[str, Any] | str) -> str:
        self.prove_nl_calls += 1
        payload = _payload_dict(gi_payload)
        self.last_prove_payload = payload
        ids = [str(node.get("id")) for node in payload.get("nodes", [])]
        body = "; ".join(
            f"{node.get('id')}: {node.get('statement', '')}"
            for node in payload.get("nodes", [])
        )
        return f"NL proof from GI nodes {', '.join(ids)}. {body}"


class MockExtractor:
    """ProofFlow-style ClaimDAG JSON. Incremental expansions merge into GA."""

    def __init__(self, *, invalid_first: bool = False) -> None:
        self.invalid_first = invalid_first
        self._extracts = 0
        self._lemma = 1
        self._dead = 0

    def extract(
        self,
        theorem: str,
        proof: str,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        self._extracts += 1
        ctx = dict(context or {})
        problem_id = str(ctx.get("problem_id") or "mock")
        if self.invalid_first and self._extracts == 1 and "existing" not in ctx:
            return json.dumps(
                {
                    "problem_id": problem_id,
                    "nodes": [
                        {
                            "id": "tc_1",
                            "natural_language": theorem,
                            "statement": theorem,
                            "dependencies": [],
                        },
                        {
                            "id": "l1",
                            "natural_language": proof,
                            "statement": proof,
                            "dependencies": ["ghost_lemma"],
                        },
                        {
                            "id": "ts_1",
                            "natural_language": theorem,
                            "statement": theorem,
                            "dependencies": ["l1"],
                        },
                    ],
                }
            )
        if "existing" in ctx:
            return self._expansion_json(theorem, proof, ctx)
        return self._writeup_json(theorem, proof, problem_id)

    def repair(
        self,
        dag_json: str,
        issues: Sequence[Any],
        context: Mapping[str, Any] | None = None,
    ) -> str:
        del issues, context
        from tot.converter.builder import fix_common_issues

        dag = loads(dag_json)
        return dumps(fix_common_issues(dag))

    def _writeup_json(self, theorem: str, proof: str, problem_id: str) -> str:
        self._lemma = max(self._lemma, 2)
        lemma = (proof or theorem or "main lemma").strip()
        dag = ClaimDAG(
            problem_id=problem_id,
            nodes=[
                ClaimNode(
                    id="tc_1",
                    natural_language=theorem,
                    statement=theorem,
                    dependencies=[],
                ),
                ClaimNode(
                    id="l1",
                    natural_language=lemma,
                    statement=lemma,
                    dependencies=["tc_1"],
                ),
                ClaimNode(
                    id="ts_1",
                    natural_language=theorem,
                    statement=theorem,
                    dependencies=["l1"],
                ),
            ],
        )
        return dumps(dag)

    def _expansion_json(self, theorem: str, proof: str, ctx: dict[str, Any]) -> str:
        existing = ctx["existing"]
        if isinstance(existing, str):
            base = loads(existing)
        else:
            base = ClaimDAG.model_validate(existing)
        last_id = ctx.get("last_claim_id") or "tc_1"
        ids = base.node_id_set()
        if last_id not in ids:
            last_id = "tc_1" if "tc_1" in ids else next(iter(ids), "tc_1")
        claim = (proof or theorem or "next claim").strip()
        dead = "dead" in claim.lower() or "unused" in claim.lower()
        if dead:
            self._dead += 1
            new_id = f"l_dead_{self._dead}"
            ts_dead = f"ts_dead_{self._dead}"
            nodes = [
                ClaimNode(
                    id=new_id,
                    natural_language=claim,
                    statement=claim,
                    dependencies=[last_id] if last_id in ids else [],
                ),
                ClaimNode(
                    id=ts_dead,
                    natural_language=claim,
                    statement=claim,
                    dependencies=[new_id],
                ),
            ]
            return dumps(ClaimDAG(problem_id=base.problem_id, nodes=nodes))

        self._lemma += 1
        new_id = f"l{self._lemma}"
        while new_id in ids:
            self._lemma += 1
            new_id = f"l{self._lemma}"
        nodes = [
            ClaimNode(
                id=new_id,
                natural_language=claim,
                statement=claim,
                dependencies=[last_id] if last_id in ids else [],
            ),
            ClaimNode(
                id="ts_1",
                natural_language=theorem,
                statement=theorem or claim,
                dependencies=[new_id],
            ),
        ]
        return dumps(ClaimDAG(problem_id=base.problem_id, nodes=nodes))


class MockProver:
    """GI-only Lean mock. Can fail a bounded number of times, then succeed."""

    def __init__(self, *, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.calls = 0
        self.last_payload: dict[str, Any] | None = None

    def prove(self, gi_payload: Mapping[str, Any] | str) -> Mapping[str, Any]:
        self.calls += 1
        payload = _payload_dict(gi_payload)
        self.last_payload = payload
        ids = [str(node.get("id")) for node in payload.get("nodes", [])]
        code = (
            "theorem gi_goal : True := by\n"
            f"  -- GI nodes: {', '.join(ids)}\n"
            "  trivial"
        )
        if self.calls <= self.fail_times:
            return {
                "ok": False,
                "code": code,
                "errors": [
                    "error: unsolved goals",
                    "⊢ False",
                    f"failed on GI nodes {', '.join(ids)}",
                ],
            }
        return {"ok": True, "code": code, "errors": []}
