"""train_dumps keep-rate protocol (no MiniLM download)."""

from __future__ import annotations

import json

import pytest

from tot.ir.dump import dump
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.route.io import load_route_meta
from tot.route.train_dumps import (
    build_arg_parser,
    main as train_dumps_main,
    pos_weight_sweep_values,
    select_keep_rate_cell,
)


def test_train_dumps_parser_tune_keep_rate_default() -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.tune_keep_rate is True
    args_off = parser.parse_args(["--no-tune-keep-rate", "--keep-threshold", "0.3"])
    assert args_off.tune_keep_rate is False
    assert args_off.keep_threshold == pytest.approx(0.3)


def test_pos_weight_sweep_dedupes_balanced_ratio() -> None:
    assert pos_weight_sweep_values(1.0) == [1.0]
    values = pos_weight_sweep_values(4.0)
    assert values == [1.0, 2.0, 4.0]


def test_select_keep_rate_prefers_zero_empty_then_closest() -> None:
    cells = [
        {
            "pos_weight": 1.0,
            "keep_threshold": 0.5,
            "empty_pred": 1,
            "nr_pred_keep_rate_pooled": 0.20,
        },
        {
            "pos_weight": 2.0,
            "keep_threshold": 0.3,
            "empty_pred": 0,
            "nr_pred_keep_rate_pooled": 0.41,
        },
        {
            "pos_weight": 2.0,
            "keep_threshold": 0.4,
            "empty_pred": 0,
            "nr_pred_keep_rate_pooled": 0.22,
        },
    ]
    best, note = select_keep_rate_cell(cells, gh_keep_rate=0.40)
    assert best["keep_threshold"] == pytest.approx(0.3)
    assert "zero empty" in note


def test_select_keep_rate_all_empty_min_then_closest() -> None:
    cells = [
        {
            "pos_weight": 1.0,
            "keep_threshold": 0.9,
            "empty_pred": 2,
            "nr_pred_keep_rate_pooled": 0.0,
        },
        {
            "pos_weight": 1.0,
            "keep_threshold": 0.8,
            "empty_pred": 1,
            "nr_pred_keep_rate_pooled": 0.10,
        },
        {
            "pos_weight": 1.0,
            "keep_threshold": 0.7,
            "empty_pred": 1,
            "nr_pred_keep_rate_pooled": 0.30,
        },
    ]
    best, note = select_keep_rate_cell(cells, gh_keep_rate=0.25)
    assert best["keep_threshold"] == pytest.approx(0.7)
    assert "min empty_pred" in note


def test_train_dumps_numpy_persists_meta(tmp_path) -> None:
    def _node(node_id: str, deps: list[str]) -> ClaimNode:
        return ClaimNode(
            id=node_id,
            natural_language=node_id,
            statement=node_id,
            dependencies=deps,
        )

    ga = ClaimDAG(
        problem_id="td_a",
        nodes=[
            _node("tc_1", []),
            _node("l1", ["tc_1"]),
            _node("l_skip", ["tc_1"]),
            _node("ts_1", ["l1", "l_skip"]),
        ],
    )
    gh = ga.induced_subgraph({"tc_1", "l1", "ts_1"})
    dump_dir = tmp_path / "td_a"
    dump_dir.mkdir()
    dump(ga, dump_dir / "ga.json")
    dump(gh, dump_dir / "gh.json")
    weights = tmp_path / "route.npz"
    assert (
        train_dumps_main(
            [
                "--dump-root",
                str(tmp_path),
                "--backend",
                "numpy",
                "--epochs",
                "1",
                "--hidden-dim",
                "4",
                "--layers",
                "1",
                "--no-tune-keep-rate",
                "--keep-threshold",
                "0.35",
                "--weights-out",
                str(weights),
            ]
        )
        == 0
    )
    meta = load_route_meta(weights)
    assert meta["keep_threshold"] == pytest.approx(0.35)
    assert meta["embedder"] == "hashed"
    assert meta["pos_weight"] is not None
    report = json.loads((tmp_path / "route_train_report.json").read_text(encoding="utf-8"))
    assert report["keep_threshold"] == pytest.approx(0.35)
    assert report["embedder"] == "hashed"
    assert "mean_always_keep_f1" in report
    assert "gh_keep_rate_pooled" in report
