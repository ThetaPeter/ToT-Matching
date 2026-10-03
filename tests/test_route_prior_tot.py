from __future__ import annotations

import ast
from pathlib import Path

from tot.eval.metrics import node_set_f1, nr_alignment_f1
from tot.route.prior import feasibility_prior, keep_set_from_scores, score_nodes_fn
from tot.route.train import train_route_network

from test_route_network import (
    GH_KEEP_IDS,
    GH_KEEP_IDS_B,
    _branching_ga,
    _branching_ga_b,
    _embeddings_a,
    _embeddings_b,
)

ROUTE_DIR = Path(__file__).resolve().parents[1] / "tot" / "route"


def test_trained_prior_keys_match_ga_and_f1_is_unit_interval() -> None:
    ga = _branching_ga()
    gb = _branching_ga_b()
    embeddings = {**_embeddings_a(), **_embeddings_b()}
    model = train_route_network(
        embeddings=embeddings,
        epochs=40,
        lr=0.2,
        hidden_dim=8,
        n_layers=1,
        seed=0,
        graphs=[(ga, set(GH_KEEP_IDS)), (gb, set(GH_KEEP_IDS_B))],
    )
    scores = feasibility_prior(model, ga, _embeddings_a())
    assert set(scores) == ga.node_id_set()
    prior_fn = score_nodes_fn(model, _embeddings_a())
    fn_scores = prior_fn(ga)
    assert set(fn_scores) == ga.node_id_set()
    keep = keep_set_from_scores(scores)
    f1 = node_set_f1(keep, GH_KEEP_IDS)
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0
    assert nr_alignment_f1(keep, GH_KEEP_IDS) == f1
    assert GH_KEEP_IDS == frozenset({"tc_1", "l1", "ts_1"})
    assert GH_KEEP_IDS_B == frozenset({"tc_a", "l_a", "ts_a"})


def test_route_does_not_import_prove_or_lean() -> None:
    forbidden = ("tot.prove", "tot.lean")
    for path in sorted(ROUTE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    assert not any(
                        name == pkg or name.startswith(pkg + ".") for pkg in forbidden
                    ), path
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not any(
                    mod == pkg or mod.startswith(pkg + ".") for pkg in forbidden
                ), path
