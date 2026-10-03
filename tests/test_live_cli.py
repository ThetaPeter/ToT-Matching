"""CLI knob resolution for live dumps (no HTTP)."""

from pathlib import Path

import pytest

from tot.config import SUPPORT_THRESHOLD, TOT_BEAM, TOT_DEPTH, TOT_K
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.live import build_arg_parser, build_route_node_prior, resolve_run_knobs
from tot.providers.http_util import extract_json_object
from tot.route.flow_dagnn import EXTRA_FEATURES, FlowDAGNN
from tot.route.io import dump_weights


def test_resolve_run_knobs_thesis_defaults() -> None:
    knobs = resolve_run_knobs(
        skip_lean_compile=False,
        lean_resumes=None,
        k=None,
        depth=None,
        beam=None,
        t=None,
    )
    assert knobs == {
        "k": TOT_K,
        "depth": TOT_DEPTH,
        "beam": TOT_BEAM,
        "t": SUPPORT_THRESHOLD,
        "lean_resumes": 1,
    }


def test_resolve_run_knobs_skip_lean_forces_zero_resumes() -> None:
    knobs = resolve_run_knobs(
        skip_lean_compile=True,
        lean_resumes=None,
        k=1,
        depth=2,
        beam=1,
        t=0.90,
    )
    assert knobs["lean_resumes"] == 0
    assert knobs["k"] == 1
    assert knobs["depth"] == 2
    assert knobs["beam"] == 1


def test_resolve_run_knobs_explicit_resumes_wins_over_skip_lean() -> None:
    knobs = resolve_run_knobs(
        skip_lean_compile=True,
        lean_resumes=2,
        k=None,
        depth=None,
        beam=None,
        t=None,
    )
    assert knobs["lean_resumes"] == 2


def test_live_parser_route_weights_and_backend() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--route-weights", "w.npz"])
    assert args.route_weights == Path("w.npz")
    assert args.backend == "numpy"
    args_torch = parser.parse_args(["--route-weights", "w.npz", "--backend", "torch"])
    assert args_torch.backend == "torch"


def test_build_route_node_prior_covers_synthetic_dag_ids(tmp_path) -> None:
    dag = ClaimDAG(
        problem_id="prior_live",
        nodes=[
            ClaimNode(
                id="tc_1",
                natural_language="goal",
                statement="goal",
                dependencies=[],
            ),
            ClaimNode(
                id="l1",
                natural_language="lemma",
                statement="lemma",
                dependencies=["tc_1"],
            ),
            ClaimNode(
                id="ts_1",
                natural_language="qed",
                statement="qed",
                dependencies=["l1"],
            ),
        ],
    )
    from tot.align.embed import SentenceEmbedder
    from tot.route.prior import embed_nodes

    embedder = SentenceEmbedder(force_hashed=True)
    emb = embed_nodes(dag, embedder)
    in_dim = int(next(iter(emb.values())).shape[0]) + EXTRA_FEATURES
    model = FlowDAGNN(in_dim=in_dim, hidden_dim=4, n_layers=1, seed=3)
    path = tmp_path / "route.npz"
    dump_weights(model, path, keep_threshold=0.4, pos_weight=1.0, embedder="hashed")
    prior, meta = build_route_node_prior(path, backend="numpy")
    scores = prior(dag)
    assert set(scores) == dag.node_id_set()
    assert all(0.0 < float(score) < 1.0 for score in scores.values())
    assert meta["keep_threshold"] == pytest.approx(0.4)
    assert meta["embedder"] == "hashed"


def test_extract_json_object_tolerates_latex_backslash() -> None:
    raw = '{"problem_id": "p", "nodes": [{"id": "l1", "statement": "\\frac{1}{2}"}]}'
    data = extract_json_object(raw)
    assert data["nodes"][0]["statement"] == r"\frac{1}{2}"
