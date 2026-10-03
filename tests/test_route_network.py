"""Route network: FlowDAGNN feasibility scores on GA, labels from GH ids."""

from __future__ import annotations

import numpy as np
import pytest

from tot.ir.dump import dump
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.route.flow_dagnn import EXTRA_FEATURES, FlowDAGNN, node_feature
from tot.route.io import dump_weights, load_route_meta, load_weights
from tot.route.prior import feasibility_prior, keep_set_from_scores
from tot.route.train import train_route_network

# Overfit GH keep-set (matched ids, not counts).
GH_KEEP_IDS = frozenset({"tc_1", "l1", "ts_1"})
GH_KEEP_IDS_B = frozenset({"tc_a", "l_a", "ts_a"})


def _node(node_id: str, deps: list[str] | None = None, **fields: object) -> ClaimNode:
    payload = {
        "id": node_id,
        "natural_language": fields.pop("natural_language", node_id),
        "statement": fields.pop("statement", node_id),
        "dependencies": list(deps or []),
    }
    payload.update(fields)
    return ClaimNode.model_validate(payload)


def _branching_ga() -> ClaimDAG:
    """``tc_1`` has two outgoing successors ``l1`` and ``l_skip``."""
    return ClaimDAG(
        problem_id="route_ga",
        nodes=[
            _node("tc_1", []),
            _node("l1", ["tc_1"]),
            _node("l_skip", ["tc_1"]),
            _node("ts_1", ["l1", "l_skip"]),
        ],
    )


def _branching_ga_b() -> ClaimDAG:
    return ClaimDAG(
        problem_id="route_ga_b",
        nodes=[
            _node("tc_a", []),
            _node("l_a", ["tc_a"]),
            _node("l_decoy", ["tc_a"]),
            _node("ts_a", ["l_a", "l_decoy"]),
        ],
    )


def _embeddings_a() -> dict[str, np.ndarray]:
    return {
        "tc_1": np.array([1.0, 0.0, 0.0, 0.1]),
        "l1": np.array([0.95, 0.05, 0.0, 0.0]),
        "l_skip": np.array([0.0, 1.0, 0.0, 0.0]),
        "ts_1": np.array([0.9, 0.0, 0.2, 0.0]),
    }


def _embeddings_b() -> dict[str, np.ndarray]:
    return {
        "tc_a": np.array([1.0, 0.05, 0.0, 0.0]),
        "l_a": np.array([0.92, 0.0, 0.1, 0.0]),
        "l_decoy": np.array([0.05, 1.0, 0.0, 0.0]),
        "ts_a": np.array([0.88, 0.0, 0.15, 0.1]),
    }


def test_flow_attention_outgoing_weights_sum_to_one() -> None:
    dag = _branching_ga()
    embeddings = _embeddings_a()
    model = FlowDAGNN(in_dim=6, hidden_dim=8, n_layers=1, seed=0)
    outgoing = model.outgoing_attention(dag, embeddings)
    assert set(outgoing["tc_1"]) == {"l1", "l_skip"}
    assert sum(outgoing["tc_1"].values()) == pytest.approx(1.0, abs=1e-6)

    ids, matrix = model.flow_attention_matrix(dag, embeddings)
    src = ids.index("tc_1")
    dsts = [ids.index("l1"), ids.index("l_skip")]
    assert matrix[src, dsts].sum() == pytest.approx(1.0, abs=1e-6)
    assert matrix[src].sum() == pytest.approx(1.0, abs=1e-6)
    rev = model.reverse_attention(dag, embeddings)
    assert sum(rev["tc_1"].values()) == pytest.approx(1.0, abs=1e-6)


def test_overfit_keep_set_matches_gh_ids(tmp_path) -> None:
    ga = _branching_ga()
    gb = _branching_ga_b()
    keep_a = set(GH_KEEP_IDS)
    keep_b = set(GH_KEEP_IDS_B)
    gh_a = ga.induced_subgraph(keep_a)
    gh_b = gb.induced_subgraph(keep_b)
    ga_path = tmp_path / "ga.json"
    gh_path = tmp_path / "gh.json"
    gb_path = tmp_path / "ga_b.json"
    ghb_path = tmp_path / "gh_b.json"
    dump(ga, ga_path)
    dump(gh_a, gh_path)
    dump(gb, gb_path)
    dump(gh_b, ghb_path)
    embeddings = {**_embeddings_a(), **_embeddings_b()}
    model = train_route_network(
        embeddings=embeddings,
        epochs=450,
        lr=0.2,
        hidden_dim=8,
        n_layers=1,
        seed=0,
        json_pairs=[(ga_path, gh_path), (gb_path, ghb_path)],
    )
    pred_a = model.predict_keep_set(ga, embeddings)
    pred_b = model.predict_keep_set(gb, embeddings)
    assert pred_a == keep_a
    assert pred_b == keep_b


def test_feasibility_prior_scores_every_ga_node() -> None:
    dag = _branching_ga()
    embeddings = _embeddings_a()
    model = FlowDAGNN(in_dim=6, hidden_dim=8, n_layers=1, seed=1)
    scores = feasibility_prior(model, dag, embeddings)
    assert set(scores) == dag.node_id_set()
    assert keep_set_from_scores(scores) <= dag.node_id_set()


def test_node_feature_support_extra_always_zero() -> None:
    embeddings = _embeddings_a()
    node = _node("tc_1", [], support=1.0, salience=0.3)
    feat = node_feature(node, embeddings)
    assert feat.shape[0] == embeddings["tc_1"].shape[0] + EXTRA_FEATURES
    assert feat[-2] == pytest.approx(0.3)
    assert feat[-1] == 0.0
    none_salience = _node("tc_1", [], support=1.0, salience=None)
    feat_none = node_feature(none_salience, embeddings)
    assert feat_none[-2] == 0.0
    assert feat_none[-1] == 0.0


def test_dump_load_weights_roundtrip(tmp_path) -> None:
    dag = _branching_ga()
    embeddings = _embeddings_a()
    model = FlowDAGNN(in_dim=6, hidden_dim=8, n_layers=1, seed=2)
    before = feasibility_prior(model, dag, embeddings)
    path = tmp_path / "route.npz"
    dump_weights(
        model,
        path,
        keep_threshold=0.3,
        pos_weight=3.5,
        embedder="hashed",
    )
    loaded = load_weights(path)
    after = feasibility_prior(loaded, dag, embeddings)
    assert set(after) == set(before) == dag.node_id_set()
    for node_id in dag.node_id_set():
        assert after[node_id] == pytest.approx(before[node_id], abs=1e-9)
    meta = load_route_meta(path)
    assert meta["keep_threshold"] == pytest.approx(0.3)
    assert meta["pos_weight"] == pytest.approx(3.5)
    assert meta["embedder"] == "hashed"
    assert meta["in_dim"] == 6
