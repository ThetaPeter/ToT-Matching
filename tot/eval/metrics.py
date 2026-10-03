"""Phase-1 metrics: Lean success, prune ratio, GI vs GH node-id F1."""

from __future__ import annotations

from collections.abc import Iterable

from tot.ir.schema import ClaimDAG
from tot.lean.prover import LeanResult


def node_set_f1(pred_ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    pred = set(pred_ids)
    gold = set(gold_ids)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def nr_alignment_f1(pred_keep: Iterable[str], gh_ids: Iterable[str]) -> float:
    """NR keep-set vs GH; same matched-set F1 as ``node_set_f1`` / alignment_f1."""
    return node_set_f1(pred_keep, gh_ids)


def always_keep_f1(ga_ids: Iterable[str], gh_ids: Iterable[str]) -> float:
    """F1 of predicting every GA node as keep vs GH (matched ids)."""
    return node_set_f1(ga_ids, gh_ids)


def keep_rate(keep_ids: Iterable[str], ga_ids: Iterable[str]) -> float:
    """``|keep ∩ ga| / |ga|``. Zero when GA is empty."""
    ga = set(ga_ids)
    if not ga:
        return 0.0
    return float(len(set(keep_ids) & ga)) / float(len(ga))


def set_precision_recall(
    pred: Iterable[str], gold: Iterable[str]
) -> tuple[float, float]:
    pred_s = set(pred)
    gold_s = set(gold)
    if not pred_s and not gold_s:
        return 1.0, 1.0
    tp = len(pred_s & gold_s)
    precision = (tp / len(pred_s)) if pred_s else 0.0
    recall = (tp / len(gold_s)) if gold_s else 0.0
    return float(precision), float(recall)


def prune_ratio(gi: ClaimDAG, ga: ClaimDAG) -> dict[str, float]:
    node_ratio = len(gi.nodes) / max(len(ga.nodes), 1)
    gi_tokens = _token_count(gi)
    ga_tokens = _token_count(ga)
    token_ratio = gi_tokens / max(ga_tokens, 1)
    return {"node_ratio": float(node_ratio), "token_ratio": float(token_ratio)}


def lean_success(result: LeanResult | bool) -> bool:
    if isinstance(result, bool):
        return result
    return bool(result.ok)


def _token_count(dag: ClaimDAG) -> int:
    total = 0
    for node in dag.nodes:
        total += len((node.statement or "").split())
        total += len((node.natural_language or "").split())
    return total
