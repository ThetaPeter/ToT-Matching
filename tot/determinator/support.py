"""Determinator support-core: vote via DAG ancestors of each search-branch sink."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

from pydantic import BaseModel, ConfigDict

from tot.config import SUPPORT_THRESHOLD
from tot.ir.schema import ClaimDAG, ClaimNode


class SearchBranch(BaseModel):
    """One root-to-leaf *search* branch (not a DAG path)."""

    model_config = ConfigDict(extra="forbid")

    branch_id: str
    sink_claim_id: str


def _coerce_branches(branches: Sequence[SearchBranch | Mapping[str, str]]) -> list[SearchBranch]:
    out: list[SearchBranch] = []
    for branch in branches:
        if isinstance(branch, SearchBranch):
            out.append(branch)
        else:
            out.append(SearchBranch.model_validate(branch))
    return out


def ancestors_including_self(dag: ClaimDAG, sink_claim_id: str) -> set[str]:
    """Walk dependencies upward from the sink. The sink depends on itself."""
    nodes = dag.node_map()
    seen: set[str] = set()
    stack = [sink_claim_id]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        node = nodes.get(current)
        if node is None:
            continue
        stack.extend(node.dependencies)
    return seen


def compute_support(
    dag: ClaimDAG,
    branches: Sequence[SearchBranch | Mapping[str, str]],
) -> dict[str, float]:
    """Fraction of search branches for which C is ancestor-or-self of the sink."""
    coerced = _coerce_branches(branches)
    ids = [node.id for node in dag.nodes]
    if not coerced:
        return {node_id: 0.0 for node_id in ids}

    counts = {node_id: 0 for node_id in ids}
    n_branches = len(coerced)
    for branch in coerced:
        depended = ancestors_including_self(dag, branch.sink_claim_id)
        for node_id in counts:
            if node_id in depended:
                counts[node_id] += 1
    return {node_id: counts[node_id] / n_branches for node_id in counts}


def support_core(
    dag: ClaimDAG,
    branches: Sequence[SearchBranch | Mapping[str, str]],
    t: float = SUPPORT_THRESHOLD,
) -> ClaimDAG:
    """Write ``support`` onto nodes; return GI = induced subgraph of nodes with support ≥ t."""
    supports = compute_support(dag, branches)
    for node in dag.nodes:
        node.support = supports[node.id]

    kept = {node.id for node in dag.nodes if supports[node.id] >= t}
    gi_nodes: list[ClaimNode] = []
    for node in dag.nodes:
        if node.id not in kept:
            continue
        gi_nodes.append(
            node.model_copy(
                update={
                    "support": supports[node.id],
                    "dependencies": [dep for dep in node.dependencies if dep in kept],
                }
            )
        )
    return ClaimDAG(problem_id=dag.problem_id, nodes=gi_nodes)
