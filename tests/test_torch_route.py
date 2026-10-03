"""Torch FlowDAGNN overfit + outgoing attention (no GAT)."""

from __future__ import annotations

import numpy as np
import pytest

from tot.ir.schema import ClaimDAG, ClaimNode
from tot.route.prior import keep_set_from_scores
from tot.route.train import train_route_network

torch = pytest.importorskip("torch")

GH_KEEP_IDS = frozenset({"tc_1", "l1", "ts_1"})
GH_KEEP_IDS_B = frozenset({"tc_a", "l_a", "ts_a"})


def _node(node_id: str, deps: list[str] | None = None) -> ClaimNode:
    return ClaimNode(
        id=node_id,
        natural_language=node_id,
        statement=node_id,
        dependencies=list(deps or []),
    )


def _ga() -> ClaimDAG:
    return ClaimDAG(
        problem_id="route_ga",
        nodes=[
            _node("tc_1", []),
            _node("l1", ["tc_1"]),
            _node("l_skip", ["tc_1"]),
            _node("ts_1", ["l1", "l_skip"]),
        ],
    )


def _gb() -> ClaimDAG:
    return ClaimDAG(
        problem_id="route_ga_b",
        nodes=[
            _node("tc_a", []),
            _node("l_a", ["tc_a"]),
            _node("l_decoy", ["tc_a"]),
            _node("ts_a", ["l_a", "l_decoy"]),
        ],
    )


def _emb_a() -> dict[str, np.ndarray]:
    return {
        "tc_1": np.array([1.0, 0.0, 0.0, 0.1]),
        "l1": np.array([0.95, 0.05, 0.0, 0.0]),
        "l_skip": np.array([0.0, 1.0, 0.0, 0.0]),
        "ts_1": np.array([0.9, 0.0, 0.2, 0.0]),
    }


def _emb_b() -> dict[str, np.ndarray]:
    return {
        "tc_a": np.array([1.0, 0.05, 0.0, 0.0]),
        "l_a": np.array([0.92, 0.0, 0.1, 0.0]),
        "l_decoy": np.array([0.05, 1.0, 0.0, 0.0]),
        "ts_a": np.array([0.88, 0.0, 0.15, 0.1]),
    }


def test_torch_outgoing_attention_sums_to_one() -> None:
    model = train_route_network(
        graphs=[(_ga(), set(GH_KEEP_IDS))],
        embeddings=[_emb_a()],
        epochs=50,
        lr=0.1,
        hidden_dim=16,
        n_layers=1,
        seed=0,
        backend="torch",
    )
    outgoing = model.outgoing_attention(_ga(), _emb_a())
    assert set(outgoing["tc_1"]) == {"l1", "l_skip"}
    assert sum(outgoing["tc_1"].values()) == pytest.approx(1.0, abs=1e-5)


def test_torch_overfit_keep_set_matches_gh_ids() -> None:
    keep_a = set(GH_KEEP_IDS)
    keep_b = set(GH_KEEP_IDS_B)
    model = train_route_network(
        graphs=[(_ga(), keep_a), (_gb(), keep_b)],
        embeddings=[_emb_a(), _emb_b()],
        epochs=400,
        lr=0.08,
        hidden_dim=16,
        n_layers=2,
        seed=0,
        backend="torch",
    )
    pred_a = keep_set_from_scores(model.score_nodes(_ga(), _emb_a()))
    pred_b = keep_set_from_scores(model.score_nodes(_gb(), _emb_b()))
    assert pred_a == keep_a
    assert pred_b == keep_b
