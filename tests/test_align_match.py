"""Synthetic GH matching: hand-crafted embeddings, no MiniLM."""

from __future__ import annotations

import numpy as np

from tot.align.match import match_assignment, match_hitter
from tot.ir.schema import ClaimDAG, ClaimNode


def _node(node_id: str, deps: list[str] | None = None) -> ClaimNode:
    return ClaimNode(
        id=node_id,
        natural_language=node_id,
        statement=node_id,
        dependencies=list(deps or []),
    )


def _chain(problem_id: str, ids: tuple[str, str, str]) -> ClaimDAG:
    a, b, c = ids
    return ClaimDAG(
        problem_id=problem_id,
        nodes=[
            _node(a, []),
            _node(b, [a]),
            _node(c, [b]),
        ],
    )


def test_obvious_bijection_gh_node_ids() -> None:
    ga = _chain("ga", ("tc_1", "l1", "ts_1"))
    gaprime = _chain("gaprime", ("tc_a", "l_a", "ts_a"))
    embeddings_a = {
        "tc_1": np.array([1.0, 0.0, 0.0, 0.0]),
        "l1": np.array([0.0, 1.0, 0.0, 0.0]),
        "ts_1": np.array([0.0, 0.0, 1.0, 0.0]),
    }
    embeddings_ap = {
        "tc_a": np.array([0.99, 0.01, 0.0, 0.0]),
        "l_a": np.array([0.01, 0.99, 0.0, 0.0]),
        "ts_a": np.array([0.01, 0.0, 0.99, 0.0]),
    }
    assignment = match_assignment(ga, gaprime, embeddings_a, embeddings_ap)
    assert assignment == {"tc_1": "tc_a", "l1": "l_a", "ts_1": "ts_a"}
    gh = match_hitter(ga, gaprime, embeddings_a, embeddings_ap)
    assert {node.id for node in gh.nodes} == {"tc_1", "l1", "ts_1"}
    assert gh.edge_set() == {("tc_1", "l1"), ("l1", "ts_1")}


def test_edge_consistency_prefers_dependency_preserving_pairing() -> None:
    """Wrong pairing l1↔ts_a, ts_1↔l_a would break tc_1→l1 vs tc_a→l_a.

    Embeddings make the swapped pairing better on cosine alone (near-orthogonal
    extra decoy dimension unused; ts_a is a decoy attractor for l1). The matcher
    must prefer the edge-consistent bijection.
    """
    ga = _chain("ga", ("tc_1", "l1", "ts_1"))
    gaprime = _chain("gaprime", ("tc_a", "l_a", "ts_a"))
    embeddings_a = {
        "tc_1": np.array([1.0, 0.0, 0.0, 0.0]),
        "l1": np.array([0.0, 1.0, 0.0, 0.0]),
        "ts_1": np.array([0.0, 0.0, 1.0, 0.0]),
    }
    embeddings_ap = {
        "tc_a": np.array([1.0, 0.0, 0.0, 0.0]),
        "l_a": np.array([0.0, 0.55, 0.55, 0.0]),
        "ts_a": np.array([0.0, 0.95, 0.05, 0.0]),
    }
    assignment = match_assignment(ga, gaprime, embeddings_a, embeddings_ap)
    assert assignment["tc_1"] == "tc_a"
    assert assignment["l1"] == "l_a"
    assert assignment["ts_1"] == "ts_a"
    assert assignment["l1"] != "ts_a"
    gh = match_hitter(ga, gaprime, embeddings_a, embeddings_ap)
    assert {node.id for node in gh.nodes} == {"tc_1", "l1", "ts_1"}
    assert gh.edge_set() == {("tc_1", "l1"), ("l1", "ts_1")}
