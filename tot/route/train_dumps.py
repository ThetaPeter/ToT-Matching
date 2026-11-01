"""Train Route Network on dumped GA/GH pairs (numpy or torch backend)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tot.align.embed import SentenceEmbedder
from tot.eval.metrics import always_keep_f1, keep_rate, nr_alignment_f1
from tot.ir.dump import load
from tot.route.flow_dagnn import KEEP_THRESHOLD, EXTRA_FEATURES, pos_weight_from_graphs
from tot.route.io import dump_weights, route_meta_payload
from tot.route.prior import embed_nodes, keep_set_from_scores
from tot.route.search_labels import (
    gold_search_path,
    load_search_tree,
    optional_path_labels,
    path_rank_lists,
    pred_path_core,
)
from tot.route.train import train_route_network

KEEP_RATE_TAU_GRID = tuple(round(i * 0.05, 2) for i in range(1, 20))


def _collect_pairs(dump_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for ga_path in sorted(dump_root.glob("*/ga.json")):
        gh_path = ga_path.with_name("gh.json")
        if gh_path.is_file():
            pairs.append((ga_path, gh_path))
    return pairs


def pos_weight_sweep_values(n_drop_over_n_keep: float) -> list[float]:
    """``{1.0, sqrt(ratio), ratio}`` with exact duplicates dropped."""
    ratio = float(n_drop_over_n_keep)
    out: list[float] = []
    for value in (1.0, float(ratio ** 0.5), ratio):
        if not any(abs(value - seen) < 1e-12 for seen in out):
            out.append(float(value))
    return out


def select_keep_rate_cell(
    cells: list[dict[str, Any]],
    gh_keep_rate: float,
) -> tuple[dict[str, Any], str]:
    """Pick (pos_weight, τ) by train keep-rate; never peeks at a test split."""
    if not cells:
        raise ValueError("empty keep-rate sweep")
    target = float(gh_keep_rate)
    zero_empty = [cell for cell in cells if int(cell["empty_pred"]) == 0]
    if zero_empty:
        best = min(
            zero_empty,
            key=lambda cell: abs(float(cell["nr_pred_keep_rate_pooled"]) - target),
        )
        note = (
            "closest node-pooled keep-rate to train GH keep-rate "
            "among runs with zero empty keep-sets"
        )
        return best, note
    best = min(
        cells,
        key=lambda cell: (
            int(cell["empty_pred"]),
            abs(float(cell["nr_pred_keep_rate_pooled"]) - target),
        ),
    )
    note = (
        "all sweep cells had empty keep-sets; "
        "min empty_pred then closest keep-rate"
    )
    return best, note


def select_f1_gap_cell(cells: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Pick (pos_weight, τ) by train F1 minus always-keep; never peeks at test."""
    if not cells:
        raise ValueError("empty F1-gap sweep")
    zero_empty = [cell for cell in cells if int(cell["empty_pred"]) == 0]
    pool = zero_empty or cells
    best = max(
        pool,
        key=lambda cell: (
            float(cell["mean_nr_vs_gh_f1"]) - float(cell["mean_always_keep_f1"]),
            float(cell["mean_nr_vs_gh_f1"]),
            -int(cell["empty_pred"]),
        ),
    )
    note = (
        "max train F1 minus always-keep among runs with zero empty keep-sets"
        if zero_empty
        else "all sweep cells had empty keep-sets; max train F1 minus always-keep"
    )
    return best, note


def _score_graphs(model, graphs, embeddings, search_feats=None) -> list[dict[str, float]]:
    scored = []
    for i, ((ga, _keep), emb) in enumerate(zip(graphs, embeddings)):
        feats = search_feats[i] if search_feats is not None else None
        scored.append(model.score_nodes(ga, emb, search_feats=feats))
    return scored


def _stats_from_path_cores(
    graphs,
    score_maps: list[dict[str, float]],
    trees: list[dict[str, Any] | None],
) -> dict[str, Any]:
    empty_pred = 0
    pred_all_ga = 0
    n_pred = 0
    n_ga = 0
    n_gold = 0
    f1s: list[float] = []
    always_f1s: list[float] = []
    rows: list[dict[str, Any]] = []
    for (ga, keep), scores, tree in zip(graphs, score_maps, trees):
        ga_ids = ga.node_id_set()
        pred = pred_path_core(tree, ga, scores) if tree else set()
        pred &= ga_ids
        n_ga += len(ga_ids)
        n_pred += len(pred)
        n_gold += len(set(keep) & ga_ids)
        if not pred:
            empty_pred += 1
        if pred == ga_ids:
            pred_all_ga += 1
        f1s.append(nr_alignment_f1(pred, keep))
        always_f1s.append(always_keep_f1(ga_ids, keep))
        rows.append(
            {
                "problem_id": ga.problem_id,
                "pred_ids": sorted(pred),
                "gold_path_ids": sorted(keep),
                "nr_vs_gh_f1": f1s[-1],
                "always_keep_f1": always_f1s[-1],
                "gh_keep_rate": keep_rate(keep, ga_ids),
                "nr_pred_keep_rate": keep_rate(pred, ga_ids),
            }
        )
    nr_pooled = (float(n_pred) / float(n_ga)) if n_ga else 0.0
    gold_pooled = (float(n_gold) / float(n_ga)) if n_ga else 0.0
    return {
        "keep_threshold": None,
        "empty_pred": empty_pred,
        "pred_all_ga": pred_all_ga,
        "gh_keep_rate_pooled": gold_pooled,
        "nr_pred_keep_rate_pooled": nr_pooled,
        "mean_nr_vs_gh_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_always_keep_f1": float(np.mean(always_f1s)) if always_f1s else 0.0,
        "rows": rows,
    }


def _stats_from_scores(
    graphs,
    score_maps: list[dict[str, float]],
    tau: float,
) -> dict[str, Any]:
    empty_pred = 0
    pred_all_ga = 0
    n_pred = 0
    n_ga = 0
    n_gh = 0
    f1s: list[float] = []
    always_f1s: list[float] = []
    rows: list[dict[str, Any]] = []
    for (ga, keep), scores in zip(graphs, score_maps):
        ga_ids = ga.node_id_set()
        pred = keep_set_from_scores(scores, threshold=float(tau))
        n_ga += len(ga_ids)
        n_pred += len(pred & ga_ids)
        n_gh += len(set(keep) & ga_ids)
        if not pred:
            empty_pred += 1
        if pred == ga_ids:
            pred_all_ga += 1
        f1s.append(nr_alignment_f1(pred, keep))
        always_f1s.append(always_keep_f1(ga_ids, keep))
        rows.append(
            {
                "problem_id": ga.problem_id,
                "pred_ids": sorted(pred),
                "gh_ids": sorted(keep),
                "nr_vs_gh_f1": f1s[-1],
                "always_keep_f1": always_f1s[-1],
                "gh_keep_rate": keep_rate(keep, ga_ids),
                "nr_pred_keep_rate": keep_rate(pred, ga_ids),
            }
        )
    nr_pooled = (float(n_pred) / float(n_ga)) if n_ga else 0.0
    gh_pooled = (float(n_gh) / float(n_ga)) if n_ga else 0.0
    return {
        "keep_threshold": float(tau),
        "empty_pred": empty_pred,
        "pred_all_ga": pred_all_ga,
        "gh_keep_rate_pooled": gh_pooled,
        "nr_pred_keep_rate_pooled": nr_pooled,
        "mean_nr_vs_gh_f1": float(np.mean(f1s)) if f1s else 0.0,
        "mean_always_keep_f1": float(np.mean(always_f1s)) if always_f1s else 0.0,
        "rows": rows,
    }


def _persist_weights(
    model,
    path: Path,
    backend: str,
    *,
    keep_threshold: float,
    pos_weight: float | None,
    embedder: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backend == "numpy":
        dump_weights(
            model,
            path,
            keep_threshold=keep_threshold,
            pos_weight=pos_weight,
            embedder=embedder,
        )
        return
    payload = model.state_dict_numpy()
    payload.update(
        route_meta_payload(
            keep_threshold=keep_threshold,
            pos_weight=pos_weight,
            embedder=embedder,
        )
    )
    np.savez(path, **payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    # live_budget dump-train/eval: --keep-threshold 0.3; --pos-weight defaults to n_drop/n_keep.
    parser.add_argument("--dump-root", type=Path, default=Path("data/dumps/valid"))
    parser.add_argument("--backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=Path("data/dumps/valid/route_weights.npz"),
    )
    parser.add_argument("--allow-minilm-download", action="store_true")
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Positive-class BCE weight when not sweeping. Default: n_drop/n_keep.",
    )
    parser.add_argument(
        "--keep-threshold",
        type=float,
        default=None,
        help=(
            "Keep-set cutoff when --no-tune-keep-rate (library default 0.5). "
            "Ignored while the keep-rate sweep is on."
        ),
    )
    parser.set_defaults(tune_keep_rate=True)
    parser.add_argument(
        "--no-tune-keep-rate",
        dest="tune_keep_rate",
        action="store_false",
        help="Disable the train-only keep-rate sweep; use --pos-weight and --keep-threshold.",
    )
    parser.set_defaults(select_f1_gap=False)
    parser.add_argument(
        "--select-f1-gap",
        dest="select_f1_gap",
        action="store_true",
        help="Sweep-select τ by train F1 minus always-keep instead of keep-rate match.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    pairs = _collect_pairs(args.dump_root)
    if not pairs:
        raise SystemExit(f"No ga.json/gh.json pairs under {args.dump_root}")

    embedder = SentenceEmbedder(allow_download=args.allow_minilm_download)
    graphs = []
    embeddings = []
    sibling_pairs = []
    search_feats = []
    path_ranks: list[tuple[list[str], list[list[str]]] | None] = []
    trees: list[dict[str, Any] | None] = []
    for i, (ga_path, gh_path) in enumerate(pairs):
        ga = load(ga_path)
        gh = load(gh_path)
        pairs_i, feats_i, keep = optional_path_labels(
            ga_path.parent, ga, gh.node_id_set()
        )
        graphs.append((ga, keep))
        emb = embed_nodes(ga, embedder)
        if i == 0 and args.allow_minilm_download and embedder.kind != "minilm":
            raise SystemExit(
                "MiniLM was requested via --allow-minilm-download but embeddings "
                "fell back to hashed vectors. Refusing to train hashed NR as the "
                "improved route network. Check that sentence-transformers and "
                "all-MiniLM-L6-v2 weights are available."
            )
        embeddings.append(emb)
        sibling_pairs.append(pairs_i)
        search_feats.append(feats_i)
        tree_path = ga_path.parent / "search_tree.json"
        if tree_path.is_file():
            tree = load_search_tree(tree_path)
            trees.append(tree)
            path_ranks.append(
                path_rank_lists(tree, gold_search_path(tree, gh.node_id_set()))
            )
        else:
            trees.append(None)
            path_ranks.append(None)
    use_path_core = all(tree is not None for tree in trees)

    first_emb = next(iter(embeddings[0].values()))
    in_dim = int(np.asarray(first_emb).reshape(-1).shape[0]) + EXTRA_FEATURES
    ratio = pos_weight_from_graphs(graphs)
    train_kwargs = dict(
        graphs=graphs,
        embeddings=embeddings,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_layers=args.layers,
        seed=args.seed,
        backend=args.backend,
        sibling_pairs=sibling_pairs,
        search_feats=search_feats,
        path_ranks=path_ranks,
    )

    sweep_table: list[dict[str, Any]] = []
    selected_note: str
    if args.tune_keep_rate:
        models: dict[float, Any] = {}
        ga_id_sets = [ga.node_id_set() for ga, _keep in graphs]
        n_ga = sum(len(ids) for ids in ga_id_sets)
        n_gh = sum(len(set(keep) & ga.node_id_set()) for (ga, keep) in graphs)
        gh_keep_rate = (float(n_gh) / float(n_ga)) if n_ga else 0.0
        for pos_weight in pos_weight_sweep_values(ratio):
            model = train_route_network(**train_kwargs, pos_weight=pos_weight)
            models[pos_weight] = model
            score_maps = _score_graphs(model, graphs, embeddings, search_feats)
            if use_path_core:
                stats = _stats_from_path_cores(graphs, score_maps, trees)
                sweep_table.append(
                    {
                        "pos_weight": float(pos_weight),
                        "keep_threshold": KEEP_THRESHOLD,
                        "nr_pred_keep_rate_pooled": stats["nr_pred_keep_rate_pooled"],
                        "empty_pred": stats["empty_pred"],
                        "pred_all_ga": stats["pred_all_ga"],
                        "mean_nr_vs_gh_f1": stats["mean_nr_vs_gh_f1"],
                        "mean_always_keep_f1": stats["mean_always_keep_f1"],
                    }
                )
            else:
                for tau in KEEP_RATE_TAU_GRID:
                    stats = _stats_from_scores(graphs, score_maps, tau)
                    sweep_table.append(
                        {
                            "pos_weight": float(pos_weight),
                            "keep_threshold": float(tau),
                            "nr_pred_keep_rate_pooled": stats["nr_pred_keep_rate_pooled"],
                            "empty_pred": stats["empty_pred"],
                            "pred_all_ga": stats["pred_all_ga"],
                            "mean_nr_vs_gh_f1": stats["mean_nr_vs_gh_f1"],
                            "mean_always_keep_f1": stats["mean_always_keep_f1"],
                        }
                    )
        if args.select_f1_gap:
            best, selected_note = select_f1_gap_cell(sweep_table)
        else:
            best, selected_note = select_keep_rate_cell(sweep_table, gh_keep_rate)
        pos_weight = float(best["pos_weight"])
        keep_threshold = float(best["keep_threshold"])
        model = min(models.items(), key=lambda item: abs(item[0] - pos_weight))[1]
    else:
        pos_weight = float(args.pos_weight) if args.pos_weight is not None else ratio
        keep_threshold = (
            float(args.keep_threshold)
            if args.keep_threshold is not None
            else KEEP_THRESHOLD
        )
        model = train_route_network(**train_kwargs, pos_weight=pos_weight)
        selected_note = (
            "no sweep; pos_weight="
            + ("CLI" if args.pos_weight is not None else "n_drop/n_keep")
            + ", keep_threshold="
            + ("CLI" if args.keep_threshold is not None else "library 0.5")
        )

    score_maps = _score_graphs(model, graphs, embeddings, search_feats)
    if use_path_core:
        stats = _stats_from_path_cores(graphs, score_maps, trees)
        keep_threshold = float(keep_threshold)
    else:
        stats = _stats_from_scores(graphs, score_maps, keep_threshold)
    report = {
        "backend": args.backend,
        "n_graphs": len(graphs),
        "embedder": embedder.kind,
        "in_dim": in_dim,
        "pos_weight": float(pos_weight),
        "keep_threshold": float(keep_threshold),
        "mean_nr_vs_gh_f1": stats["mean_nr_vs_gh_f1"],
        "mean_always_keep_f1": stats["mean_always_keep_f1"],
        "gh_keep_rate_pooled": stats["gh_keep_rate_pooled"],
        "nr_pred_keep_rate_pooled": stats["nr_pred_keep_rate_pooled"],
        "empty_pred": stats["empty_pred"],
        "pred_all_ga": stats["pred_all_ga"],
        "sweep": sweep_table,
        "selected_criterion": selected_note,
        "rows": stats["rows"],
    }
    report_path = args.dump_root / "route_train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _persist_weights(
        model,
        args.weights_out,
        args.backend,
        keep_threshold=float(keep_threshold),
        pos_weight=float(pos_weight),
        embedder=embedder.kind,
    )
    print(
        f"backend={args.backend} n={len(graphs)} "
        f"mean NR-vs-GH F1={report['mean_nr_vs_gh_f1']:.3f} "
        f"always-keep={report['mean_always_keep_f1']:.3f} "
        f"GH-keep={report['gh_keep_rate_pooled']:.3f} "
        f"NR-keep={report['nr_pred_keep_rate_pooled']:.3f} "
        f"pos_weight={pos_weight:.4g} keep_threshold={keep_threshold}"
    )
    print(f"Wrote {args.weights_out} and {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
