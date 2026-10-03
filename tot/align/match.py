"""Hungarian matching of GA to GA' with a simple edge-consistency term."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from tot.ir.schema import ClaimDAG

DEFAULT_EDGE_CONSISTENCY_WEIGHT = 1.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=float).reshape(-1)
    right = np.asarray(b, dtype=float).reshape(-1)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def _require_embeddings(
    dag: ClaimDAG,
    embeddings: Mapping[str, np.ndarray],
    label: str,
) -> dict[str, np.ndarray]:
    missing = [node.id for node in dag.nodes if node.id not in embeddings]
    if missing:
        raise KeyError(f"missing embeddings for {label} nodes: {missing}")
    return {node.id: np.asarray(embeddings[node.id], dtype=float) for node in dag.nodes}


def _predecessors(dag: ClaimDAG) -> dict[str, list[str]]:
    return {node.id: list(node.dependencies) for node in dag.nodes}


def _successors(dag: ClaimDAG) -> dict[str, list[str]]:
    succ = {node.id: [] for node in dag.nodes}
    for node in dag.nodes:
        for dep in node.dependencies:
            if dep in succ:
                succ[dep].append(node.id)
    return succ


def _neighbor_bonus(
    src_ids: list[str],
    dst_ids: list[str],
    sim: dict[tuple[str, str], float],
) -> float:
    if not src_ids or not dst_ids:
        return 0.0
    total = 0.0
    for src in src_ids:
        total += max(sim.get((src, dst), 0.0) for dst in dst_ids)
    return total / max(len(src_ids), len(dst_ids))


def _pair_scores(
    ga: ClaimDAG,
    gaprime: ClaimDAG,
    sim: dict[tuple[str, str], float],
    edge_consistency_weight: float,
) -> dict[tuple[str, str], float]:
    pred_a = _predecessors(ga)
    pred_p = _predecessors(gaprime)
    succ_a = _successors(ga)
    succ_p = _successors(gaprime)
    scores: dict[tuple[str, str], float] = {}
    for u in pred_a:
        for up in pred_p:
            pred_b = _neighbor_bonus(pred_a[u], pred_p[up], sim)
            succ_b = _neighbor_bonus(succ_a[u], succ_p[up], sim)
            structural = 0.5 * (pred_b + succ_b)
            scores[u, up] = sim[u, up] + edge_consistency_weight * structural
    return scores


def _assignment_score(
    mapping: dict[str, str],
    ga_edges: set[tuple[str, str]],
    gp_edges: set[tuple[str, str]],
    sim: dict[tuple[str, str], float],
    edge_consistency_weight: float,
) -> float:
    sim_sum = sum(sim[u, mapping[u]] for u in mapping)
    preserved = 0
    for u, v in ga_edges:
        if u in mapping and v in mapping and (mapping[u], mapping[v]) in gp_edges:
            preserved += 1
    return sim_sum + edge_consistency_weight * preserved


def _refine_edge_consistency(
    mapping: dict[str, str],
    ga: ClaimDAG,
    gaprime: ClaimDAG,
    sim: dict[tuple[str, str], float],
    edge_consistency_weight: float,
) -> dict[str, str]:
    if len(mapping) < 2:
        return dict(mapping)
    ga_edges = ga.edge_set()
    gp_edges = gaprime.edge_set()
    current = dict(mapping)
    improved = True
    while improved:
        improved = False
        best = _assignment_score(current, ga_edges, gp_edges, sim, edge_consistency_weight)
        keys = list(current.keys())
        for i, j in combinations(range(len(keys)), 2):
            u, v = keys[i], keys[j]
            trial = dict(current)
            trial[u], trial[v] = trial[v], trial[u]
            score = _assignment_score(trial, ga_edges, gp_edges, sim, edge_consistency_weight)
            if score > best + 1e-12:
                current = trial
                best = score
                improved = True
                break
    return current


def match_assignment(
    ga: ClaimDAG,
    gaprime: ClaimDAG,
    embeddings_a: Mapping[str, np.ndarray],
    embeddings_ap: Mapping[str, np.ndarray],
    *,
    edge_consistency_weight: float = DEFAULT_EDGE_CONSISTENCY_WEIGHT,
    min_similarity: float | None = None,
) -> dict[str, str]:
    """Map GA node ids → GA′ node ids (unmatched GA nodes omitted)."""
    if not ga.nodes or not gaprime.nodes:
        return {}

    vecs_a = _require_embeddings(ga, embeddings_a, "GA")
    vecs_p = _require_embeddings(gaprime, embeddings_ap, "GA'")
    ga_ids = [node.id for node in ga.nodes]
    gp_ids = [node.id for node in gaprime.nodes]

    sim: dict[tuple[str, str], float] = {}
    for u in ga_ids:
        for up in gp_ids:
            sim[u, up] = cosine_similarity(vecs_a[u], vecs_p[up])

    pair_scores = _pair_scores(ga, gaprime, sim, edge_consistency_weight)
    n, m = len(ga_ids), len(gp_ids)
    cost = np.empty((n, m), dtype=float)
    for i, u in enumerate(ga_ids):
        for j, up in enumerate(gp_ids):
            cost[i, j] = -pair_scores[u, up]

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {ga_ids[i]: gp_ids[j] for i, j in zip(row_ind, col_ind)}
    mapping = _refine_edge_consistency(
        mapping, ga, gaprime, sim, edge_consistency_weight
    )
    if min_similarity is not None:
        mapping = {
            u: up for u, up in mapping.items() if sim[u, up] >= min_similarity
        }
    return mapping


def match_hitter(
    ga: ClaimDAG,
    gaprime: ClaimDAG,
    embeddings_a: Mapping[str, np.ndarray],
    embeddings_ap: Mapping[str, np.ndarray],
    *,
    edge_consistency_weight: float = DEFAULT_EDGE_CONSISTENCY_WEIGHT,
    min_similarity: float | None = None,
) -> ClaimDAG:
    """GH: induced subgraph of GA on nodes that matched a GA′ node."""
    mapping = match_assignment(
        ga,
        gaprime,
        embeddings_a,
        embeddings_ap,
        edge_consistency_weight=edge_consistency_weight,
        min_similarity=min_similarity,
    )
    return ga.induced_subgraph(mapping.keys())
