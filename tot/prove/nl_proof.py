"""GI-only natural-language proof. Never serialize the search tree or extra GA nodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tot.config import SUPPORT_THRESHOLD
from tot.determinator.support import SearchBranch, support_core
from tot.ir.schema import ClaimDAG
from tot.providers.base import Reasoner


def serialize_gi(gi: ClaimDAG) -> dict[str, Any]:
    """Serialize GI only: node id, statement, dependencies."""
    return {
        "problem_id": gi.problem_id,
        "nodes": [
            {
                "id": node.id,
                "statement": node.statement,
                "dependencies": list(node.dependencies),
            }
            for node in gi.nodes
        ],
    }


def gi_payload_ids(payload: Mapping[str, Any]) -> set[str]:
    return {str(node["id"]) for node in payload.get("nodes", [])}


def build_gi_payload(
    ga: ClaimDAG,
    branches: Sequence[SearchBranch | Mapping[str, str]],
    t: float = SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    """From a GA that may have extra nodes, emit a payload whose ids are GI only."""
    gi = support_core(ga, branches, t=t)
    return serialize_gi(gi)


def prove_nl(reasoner: Reasoner, gi: ClaimDAG) -> str:
    payload = serialize_gi(gi)
    return reasoner.prove_nl(payload)
