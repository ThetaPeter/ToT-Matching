"""Held-out eval report (slice D).

Proof-quality metric (locked for D): token-F1 and MiniLM cosine vs human writeup.
GI-vs-GH and NR-vs-GH use matched node-id sets (not counts).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tot.align.embed import SentenceEmbedder
from tot.eval.metrics import (
    always_keep_f1,
    keep_rate,
    lean_success,
    node_set_f1,
    nr_alignment_f1,
    prune_ratio,
    set_precision_recall,
)
from tot.ir.dump import load
from tot.route.flow_dagnn import EXTRA_FEATURES, KEEP_THRESHOLD
from tot.route.io import load_route_meta, load_route_model
from tot.route.prior import embed_nodes, embedder_for_route_weights, keep_set_from_scores
from tot.route.search_labels import (
    keep_set_from_sibling_rank,
    load_search_tree,
    optional_path_labels,
    optional_tree_labels,
    pred_path_core,
)


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = pred.lower().split()
    gold_toks = gold.lower().split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    pred_set = set(pred_toks)
    gold_set = set(gold_toks)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pooled_keep_rate(keep_sets: list[set[str]], ga_sets: list[set[str]]) -> float | None:
    n_keep = 0
    n_ga = 0
    for keep, ga in zip(keep_sets, ga_sets):
        n_ga += len(ga)
        n_keep += len(keep & ga)
    if n_ga <= 0:
        return None
    return float(n_keep) / float(n_ga)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # live_budget dump-train/eval: --keep-threshold 0.3 (KEEP_THRESHOLD library default stays 0.5).
    parser.add_argument("--dump-root", type=Path, default=Path("data/dumps/test"))
    parser.add_argument(
        "--writeups",
        type=Path,
        default=None,
        help="Optional JSON map {problem_id: human_proof}",
    )
    parser.add_argument("--route-weights", type=Path, default=None)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="torch")
    parser.add_argument("--allow-minilm-download", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--keep-threshold",
        type=float,
        default=None,
        help=(
            "Keep-set cutoff on NR scores. Explicit CLI wins; else persisted τ "
            "from --route-weights; else library default 0.5."
        ),
    )
    args = parser.parse_args(argv)

    writeups: dict[str, str] = {}
    if args.writeups and args.writeups.is_file():
        writeups = json.loads(args.writeups.read_text(encoding="utf-8"))

    embedder = SentenceEmbedder(allow_download=args.allow_minilm_download)
    route_model = None
    route_meta: dict[str, object] = {}
    keep_threshold = args.keep_threshold
    if args.route_weights and args.route_weights.is_file():
        route_meta = load_route_meta(args.route_weights)
        route_model = load_route_model(args.route_weights, args.backend)
        embedder = embedder_for_route_weights(
            int(route_model.in_dim),
            persisted_kind=str(route_meta.get("embedder") or ""),
            allow_download=args.allow_minilm_download,
        )
        probe = embedder.embed(["route-weights dim probe"])
        expected_emb = int(route_model.in_dim) - EXTRA_FEATURES
        if int(probe.shape[1]) != expected_emb:
            raise SystemExit(
                f"route-weights in_dim={route_model.in_dim} expects embed dim "
                f"{expected_emb} but {embedder.kind} embedder produced {int(probe.shape[1])}."
            )
        persisted = route_meta.get("keep_threshold")
        if keep_threshold is None and persisted is not None:
            keep_threshold = float(persisted)
    if keep_threshold is None:
        keep_threshold = KEEP_THRESHOLD

    rows: list[dict[str, object]] = []
    ga_sets: list[set[str]] = []
    gh_sets: list[set[str]] = []
    gi_sets: list[set[str]] = []
    nr_sets: list[set[str]] = []
    for metrics_path in sorted(args.dump_root.glob("*/metrics.json")):
        problem_dir = metrics_path.parent
        problem_id = problem_dir.name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        ga = load(problem_dir / "ga.json")
        gh = load(problem_dir / "gh.json")
        gi = load(problem_dir / "gi.json")
        nl_path = problem_dir / "nl_proof.txt"
        nl_proof = nl_path.read_text(encoding="utf-8") if nl_path.is_file() else ""

        lean_flag = metrics.get("lean_success")
        if lean_flag is None:
            lean_flag = False
        ga_ids = ga.node_id_set()
        gh_ids = gh.node_id_set()
        gi_ids = gi.node_id_set()
        ga_sets.append(ga_ids)
        gh_sets.append(gh_ids)
        gi_sets.append(gi_ids)
        row: dict[str, object] = {
            "id": problem_id,
            "lean_success": bool(lean_success(bool(lean_flag))),
            "prune_ratio": metrics.get("prune_ratio") or prune_ratio(gi, ga),
            "gi_vs_gh_f1": float(
                metrics["alignment_f1"]
                if "alignment_f1" in metrics
                else node_set_f1(gi_ids, gh_ids)
            ),
            "gi_ids": sorted(gi_ids),
            "gh_ids": sorted(gh_ids),
        }

        if route_model is not None:
            emb = embed_nodes(ga, embedder)
            tree_path = problem_dir / "search_tree.json"
            if tree_path.is_file():
                _pairs, search_feats, gold_core = optional_path_labels(
                    problem_dir, ga, gh_ids
                )
                scores = route_model.score_nodes(ga, emb, search_feats=search_feats)
                tree = load_search_tree(tree_path)
                pred = pred_path_core(tree, ga, scores) & ga_ids
                gold = gold_core & ga_ids
                row["gold_path_ids"] = sorted(gold)
            else:
                _pairs, search_feats = optional_tree_labels(problem_dir, gh_ids)
                scores = route_model.score_nodes(ga, emb, search_feats=search_feats)
                pred = keep_set_from_scores(scores, threshold=float(keep_threshold))
                gold = gh_ids
            nr_precision, nr_recall = set_precision_recall(pred, gold)
            row["nr_pred_ids"] = sorted(pred)
            row["nr_vs_gh_f1"] = nr_alignment_f1(pred, gold)
            row["always_keep_f1"] = always_keep_f1(ga_ids, gold)
            row["gh_keep_rate"] = keep_rate(gold, ga_ids)
            row["nr_pred_keep_rate"] = keep_rate(pred, ga_ids)
            row["gi_keep_rate"] = keep_rate(gi_ids, ga_ids)
            row["nr_precision"] = nr_precision
            row["nr_recall"] = nr_recall
            row["nr_pred_is_all_ga"] = pred == ga_ids
            row["nr_pred_empty"] = len(pred) == 0
            nr_sets.append(pred)
            gh_sets[-1] = gold
            if tree_path.is_file():
                sib = keep_set_from_sibling_rank(load_search_tree(tree_path), scores)
                sib &= ga_ids
                if sib:
                    sib_p, sib_r = set_precision_recall(sib, gold)
                    row["sibling_pred_ids"] = sorted(sib)
                    row["sibling_vs_gh_f1"] = nr_alignment_f1(sib, gold)
                    row["sibling_keep_rate"] = keep_rate(sib, ga_ids)
                    row["sibling_precision"] = sib_p
                    row["sibling_recall"] = sib_r

        gold = writeups.get(problem_id)
        if gold is not None and nl_proof:
            row["proof_token_f1"] = _token_f1(nl_proof, gold)
            mats = embedder.embed([nl_proof, gold])
            row["proof_embedding_cosine"] = _cosine(
                np.asarray(mats[0], dtype=np.float64),
                np.asarray(mats[1], dtype=np.float64),
            )

        rows.append(row)

    def _mean(key: str) -> float | None:
        vals = [float(row[key]) for row in rows if key in row]
        return float(np.mean(vals)) if vals else None

    empty_pred_count = sum(1 for row in rows if row.get("nr_pred_empty") is True)
    pred_all_ga_count = sum(1 for row in rows if row.get("nr_pred_is_all_ga") is True)
    report: dict[str, object] = {
        "n": len(rows),
        "lean_success_rate": _mean("lean_success"),
        "mean_gi_vs_gh_f1": _mean("gi_vs_gh_f1"),
        "mean_nr_vs_gh_f1": _mean("nr_vs_gh_f1"),
        "mean_always_keep_f1": _mean("always_keep_f1"),
        "mean_proof_token_f1": _mean("proof_token_f1"),
        "mean_proof_embedding_cosine": _mean("proof_embedding_cosine"),
        "gh_keep_rate_pooled": _pooled_keep_rate(gh_sets, ga_sets),
        "gi_keep_rate_pooled": _pooled_keep_rate(gi_sets, ga_sets),
        "nr_pred_keep_rate_pooled": (
            _pooled_keep_rate(nr_sets, ga_sets) if nr_sets else None
        ),
        "empty_pred_count": empty_pred_count if route_model is not None else None,
        "pred_all_ga_count": pred_all_ga_count if route_model is not None else None,
        "mean_nr_precision": _mean("nr_precision"),
        "mean_nr_recall": _mean("nr_recall"),
        "mean_sibling_vs_gh_f1": _mean("sibling_vs_gh_f1"),
        "mean_sibling_keep_rate": _mean("sibling_keep_rate"),
        "mean_sibling_precision": _mean("sibling_precision"),
        "mean_sibling_recall": _mean("sibling_recall"),
        "keep_threshold_used": float(keep_threshold) if route_model is not None else None,
        "embedder": embedder.kind,
        "proof_quality_metric": "token_f1 + MiniLM cosine vs human writeup",
        "rows": rows,
    }
    persisted_pw = route_meta.get("pos_weight") if route_meta else None
    if persisted_pw is not None:
        report["pos_weight"] = float(persisted_pw)
    persisted_embedder = route_meta.get("embedder") if route_meta else None
    if persisted_embedder:
        report["embedder"] = str(persisted_embedder)
    out = args.out or (args.dump_root / "eval_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"n={report['n']} lean={report['lean_success_rate']} "
        f"GI-vs-GH={report['mean_gi_vs_gh_f1']} "
        f"NR-vs-gold={report['mean_nr_vs_gh_f1']} "
        f"always-keep={report['mean_always_keep_f1']} "
        f"gold-keep={report['gh_keep_rate_pooled']} "
        f"NR-keep={report['nr_pred_keep_rate_pooled']} "
        f"GI-keep={report['gi_keep_rate_pooled']} "
        f"NR-P={report['mean_nr_precision']} "
        f"NR-R={report['mean_nr_recall']} "
        f"sibling-vs-gold={report['mean_sibling_vs_gh_f1']} "
        f"sibling-keep={report['mean_sibling_keep_rate']} "
        f"proof_token_f1={report['mean_proof_token_f1']}"
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
