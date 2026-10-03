from __future__ import annotations

import json

import pytest

from tot.eval.metrics import (
    always_keep_f1,
    keep_rate,
    lean_success,
    node_set_f1,
    nr_alignment_f1,
    prune_ratio,
    set_precision_recall,
)
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.lean.prover import LeanResult


def _dag(problem_id: str, ids: list[str]) -> ClaimDAG:
    nodes: list[ClaimNode] = []
    for i, node_id in enumerate(ids):
        deps = [ids[i - 1]] if i else []
        nodes.append(
            ClaimNode(
                id=node_id,
                natural_language=node_id,
                statement=f"{node_id} statement tokens here",
                dependencies=deps,
            )
        )
    return ClaimDAG(problem_id=problem_id, nodes=nodes)


def test_node_set_f1_matched_sets() -> None:
    gi = {"tc_1", "l1", "ts_1"}
    gh = {"tc_1", "l1", "ts_1"}
    assert node_set_f1(gi, gh) == 1.0
    pred = {"tc_1", "l1"}
    gold = {"tc_1", "l1", "ts_1"}
    assert node_set_f1(pred, gold) == pytest.approx(0.8)
    assert node_set_f1(set(), {"tc_1"}) == 0.0
    assert node_set_f1(set(), set()) == 1.0
    assert node_set_f1({"l_dead"}, gi) == 0.0
    assert nr_alignment_f1(gi, gh) == node_set_f1(gi, gh)


def test_always_keep_keep_rate_precision_recall_matched_sets() -> None:
    ga = {"tc_1", "l1", "l_skip", "ts_1"}
    gh = {"tc_1", "l1", "ts_1"}
    assert always_keep_f1(ga, gh) == node_set_f1(ga, gh)
    assert keep_rate(gh, ga) == pytest.approx(0.75)
    assert keep_rate(set(), ga) == 0.0
    assert keep_rate(gh, set()) == 0.0
    precision, recall = set_precision_recall(gh, gh)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    precision, recall = set_precision_recall({"tc_1", "l1"}, gh)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2.0 / 3.0)
    precision, recall = set_precision_recall(set(), set())
    assert (precision, recall) == (1.0, 1.0)
    precision, recall = set_precision_recall(set(), gh)
    assert (precision, recall) == (0.0, 0.0)


def test_prune_ratio_and_lean_success() -> None:
    ga = _dag("ga", ["tc_1", "l1", "l_dead", "ts_1"])
    gi = _dag("gi", ["tc_1", "l1", "ts_1"])
    ratios = prune_ratio(gi, ga)
    assert ratios["node_ratio"] == pytest.approx(0.75)
    assert 0.0 < ratios["token_ratio"] <= 1.0
    ok = LeanResult(ok=True, code="trivial", errors=[], new_proof_state="")
    bad = LeanResult(ok=False, code="", errors=["fail"], new_proof_state="resume")
    assert lean_success(ok) is True
    assert lean_success(bad) is False
    assert lean_success(True) is True


def test_report_cli_uses_persisted_keep_threshold(tmp_path) -> None:
    from tot.align.embed import SentenceEmbedder
    from tot.eval.report import main as report_main
    from tot.ir.dump import dump
    from tot.route.flow_dagnn import EXTRA_FEATURES, FlowDAGNN
    from tot.route.io import dump_weights
    from tot.route.prior import embed_nodes

    problem_dir = tmp_path / "prob_a"
    problem_dir.mkdir()
    ga = _dag("prob_a", ["tc_1", "l1", "l_skip", "ts_1"])
    gh = _dag("prob_a", ["tc_1", "l1", "ts_1"])
    gi = _dag("prob_a", ["tc_1", "l1", "ts_1"])
    dump(ga, problem_dir / "ga.json")
    dump(gh, problem_dir / "gh.json")
    dump(gi, problem_dir / "gi.json")
    (problem_dir / "metrics.json").write_text(
        json.dumps({"lean_success": False}), encoding="utf-8"
    )
    embedder = SentenceEmbedder(force_hashed=True)
    emb = embed_nodes(ga, embedder)
    in_dim = int(next(iter(emb.values())).shape[0]) + EXTRA_FEATURES
    model = FlowDAGNN(in_dim=in_dim, hidden_dim=4, n_layers=1, seed=0)
    weights = tmp_path / "route.npz"
    dump_weights(
        model, weights, keep_threshold=0.3, pos_weight=2.5, embedder="hashed"
    )
    out = tmp_path / "eval_report.json"
    assert report_main(
        [
            "--dump-root",
            str(tmp_path),
            "--route-weights",
            str(weights),
            "--backend",
            "numpy",
            "--out",
            str(out),
        ]
    ) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["keep_threshold_used"] == pytest.approx(0.3)
    assert report["embedder"] == "hashed"
    assert report["pos_weight"] == pytest.approx(2.5)
    assert report["rows"][0]["always_keep_f1"] == always_keep_f1(
        ga.node_id_set(), gh.node_id_set()
    )
    assert "nr_pred_ids" in report["rows"][0]
    assert "nr_pred_empty" in report["rows"][0]
