"""Search-tree labels for the route network: sibling rank pairs and extras."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tot.config import TOT_DEPTH
from tot.determinator.support import ancestors_including_self
from tot.ir.schema import ClaimDAG

SEARCH_DEPTH_NORM = float(TOT_DEPTH)
ClaimPair = tuple[str, str]
SearchFeats = dict[str, tuple[float, float]]


def load_search_tree(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def claim_search_features(tree: Mapping[str, Any]) -> SearchFeats:
    """Map claim id → (depth / 8, reasoner score). Shallower write wins."""
    feats: SearchFeats = {}
    nodes = list(tree.get("nodes") or [])
    nodes.sort(key=lambda node: (int(node.get("depth") or 0), str(node.get("id") or "")))
    for node in nodes:
        claim_id = node.get("last_claim_id")
        if not claim_id or claim_id in feats:
            continue
        depth = float(node.get("depth") or 0) / SEARCH_DEPTH_NORM
        score = float(node.get("score") or 0.0)
        feats[str(claim_id)] = (depth, score)
    return feats


def _index_tree(
    tree: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    nodes = {
        str(node["id"]): node
        for node in (tree.get("nodes") or [])
        if node.get("id")
    }
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        parent_id = node.get("parent_id")
        if parent_id:
            children[str(parent_id)].append(node_id)
    return nodes, children


def _reachable_claims(
    node_id: str,
    nodes: Mapping[str, Mapping[str, Any]],
    children: Mapping[str, list[str]],
    cache: dict[str, set[str]],
) -> set[str]:
    if node_id in cache:
        return cache[node_id]
    node = nodes[node_id]
    found: set[str] = set()
    claim_id = node.get("last_claim_id")
    if claim_id:
        found.add(str(claim_id))
    for child_id in children.get(node_id, []):
        found |= _reachable_claims(child_id, nodes, children, cache)
    cache[node_id] = found
    return found


def _close_ancestors(dag: ClaimDAG, claim_ids: Iterable[str]) -> set[str]:
    kept: set[str] = set()
    ga_ids = dag.node_id_set()
    for claim_id in claim_ids:
        if claim_id not in ga_ids:
            continue
        kept |= ancestors_including_self(dag, str(claim_id))
    return kept & ga_ids


def _path_claim_ids(
    nodes: Mapping[str, Mapping[str, Any]],
    sn_ids: Sequence[str],
) -> list[str]:
    claims: list[str] = []
    seen: set[str] = set()
    for sn_id in sn_ids:
        claim_id = nodes.get(sn_id, {}).get("last_claim_id")
        if not claim_id:
            continue
        cid = str(claim_id)
        if cid in seen:
            continue
        seen.add(cid)
        claims.append(cid)
    return claims


def _walk_to_root(
    nodes: Mapping[str, Mapping[str, Any]],
    node_id: str,
) -> list[str]:
    path: list[str] = []
    current: str | None = node_id
    seen: set[str] = set()
    while current and current in nodes and current not in seen:
        seen.add(current)
        path.append(current)
        parent_id = nodes[current].get("parent_id")
        current = str(parent_id) if parent_id else None
    path.reverse()
    return path


def _root_ids(nodes: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted(
        node_id for node_id, node in nodes.items() if not node.get("parent_id")
    )


def leaf_sn_paths(tree: Mapping[str, Any]) -> list[list[str]]:
    """Every root-to-leaf search-node path (small ToT trees)."""
    nodes, children = _index_tree(tree)
    roots = _root_ids(nodes)
    if not roots:
        return []
    out: list[list[str]] = []

    def dfs(node_id: str, acc: list[str]) -> None:
        kids = [kid for kid in children.get(node_id, []) if kid not in acc]
        if not kids:
            out.append(acc)
            return
        for kid in kids:
            dfs(kid, acc + [kid])

    dfs(roots[0], [roots[0]])
    return out


def leaf_claim_paths(tree: Mapping[str, Any]) -> list[list[str]]:
    nodes, _children = _index_tree(tree)
    return [_path_claim_ids(nodes, path) for path in leaf_sn_paths(tree)]


def _path_score_sum(
    nodes: Mapping[str, Mapping[str, Any]],
    sn_path: Sequence[str],
    scores: Mapping[str, float],
) -> float:
    total = 0.0
    for sn_id in sn_path:
        claim_id = nodes.get(sn_id, {}).get("last_claim_id")
        if not claim_id:
            continue
        total += float(scores.get(str(claim_id), 0.0))
    return total


def gold_search_path(
    tree: Mapping[str, Any],
    gh_ids: Iterable[str],
) -> list[str]:
    """Search-node ids from root to the leaf whose path covers the most of GH."""
    gold = {str(node_id) for node_id in gh_ids}
    nodes, children = _index_tree(tree)
    if not nodes:
        return []
    leaves = [node_id for node_id in nodes if not children.get(node_id)]
    if not leaves:
        leaves = list(nodes)
    best_id = ""
    best_key = (-1, -1, "")
    for node_id in leaves:
        path = _walk_to_root(nodes, node_id)
        overlap = len(set(_path_claim_ids(nodes, path)) & gold)
        key = (overlap, int(nodes[node_id].get("depth") or 0), node_id)
        if key > best_key:
            best_key = key
            best_id = node_id
    if best_key[0] <= 0:
        roots = _root_ids(nodes)
        return [roots[0]] if roots else [best_id] if best_id else []
    return _walk_to_root(nodes, best_id)


def gold_path_core(
    tree: Mapping[str, Any],
    dag: ClaimDAG,
    gh_ids: Iterable[str],
) -> tuple[list[str], set[str]]:
    """Gold search-node path and claim-id core (path + DAG ancestors)."""
    sn_path = gold_search_path(tree, gh_ids)
    nodes, _children = _index_tree(tree)
    claims = _path_claim_ids(nodes, sn_path)
    return sn_path, _close_ancestors(dag, claims)


def pred_search_path(
    tree: Mapping[str, Any],
    scores: Mapping[str, float],
) -> list[str]:
    """Pick the root-to-leaf path with the largest sum of NR claim scores."""
    nodes, _children = _index_tree(tree)
    paths = leaf_sn_paths(tree)
    if not paths:
        return []
    best = paths[0]
    best_key = (float("-inf"), -1, "")
    for path in paths:
        leaf_id = path[-1]
        key = (
            _path_score_sum(nodes, path, scores),
            int(nodes.get(leaf_id, {}).get("depth") or 0),
            leaf_id,
        )
        if key > best_key:
            best_key = key
            best = path
    return best


def pred_path_core(
    tree: Mapping[str, Any],
    dag: ClaimDAG,
    scores: Mapping[str, float],
) -> set[str]:
    """Predicted claim-id core: NR winning path plus DAG ancestors."""
    nodes, _children = _index_tree(tree)
    claims = _path_claim_ids(nodes, pred_search_path(tree, scores))
    return _close_ancestors(dag, claims)


def sibling_rank_pairs(
    tree: Mapping[str, Any],
    gh_ids: Iterable[str],
) -> list[ClaimPair]:
    """(pos, neg) claim ids: mixed sibling groups; skip all-pos / all-neg."""
    gold = {str(node_id) for node_id in gh_ids}
    nodes, children = _index_tree(tree)
    cache: dict[str, set[str]] = {}
    pairs: list[ClaimPair] = []
    for kid_ids in children.values():
        positives: list[str] = []
        negatives: list[str] = []
        for kid_id in kid_ids:
            claim_id = nodes[kid_id].get("last_claim_id")
            if not claim_id:
                continue
            hits = bool(_reachable_claims(kid_id, nodes, children, cache) & gold)
            bucket = positives if hits else negatives
            bucket.append(str(claim_id))
        if not positives or not negatives:
            continue
        for pos in positives:
            for neg in negatives:
                if pos != neg:
                    pairs.append((pos, neg))
    return pairs


def path_rank_lists(
    tree: Mapping[str, Any],
    gold_sn_ids: Sequence[str],
) -> tuple[list[str], list[list[str]]]:
    """Gold path claim ids and the other root-to-leaf claim paths."""
    nodes, _children = _index_tree(tree)
    gold_sn = [str(node_id) for node_id in gold_sn_ids]
    gold_claims = _path_claim_ids(nodes, gold_sn)
    others = [
        _path_claim_ids(nodes, path)
        for path in leaf_sn_paths(tree)
        if path != gold_sn
    ]
    return gold_claims, others


def sibling_rank_pairs_on_path(
    tree: Mapping[str, Any],
    gold_sn_ids: Iterable[str],
) -> list[ClaimPair]:
    """Positive child iff its search node lies on the gold path."""
    on_path = {str(node_id) for node_id in gold_sn_ids}
    nodes, children = _index_tree(tree)
    pairs: list[ClaimPair] = []
    for kid_ids in children.values():
        positives: list[str] = []
        negatives: list[str] = []
        for kid_id in kid_ids:
            claim_id = nodes[kid_id].get("last_claim_id")
            if not claim_id:
                continue
            bucket = positives if kid_id in on_path else negatives
            bucket.append(str(claim_id))
        if not positives or not negatives:
            continue
        for pos in positives:
            for neg in negatives:
                if pos != neg:
                    pairs.append((pos, neg))
    return pairs


def optional_tree_labels(
    dump_dir: Path,
    gh_ids: Iterable[str],
) -> tuple[list[ClaimPair], SearchFeats]:
    """Load sibling pairs and claim extras from ``search_tree.json`` if present."""
    path = dump_dir / "search_tree.json"
    if not path.is_file():
        return [], {}
    tree = load_search_tree(path)
    return sibling_rank_pairs(tree, gh_ids), claim_search_features(tree)


def optional_path_labels(
    dump_dir: Path,
    dag: ClaimDAG,
    gh_ids: Iterable[str],
) -> tuple[list[ClaimPair], SearchFeats, set[str]]:
    """Path-core BCE labels and on-path sibling pairs, or GH if no tree."""
    path = dump_dir / "search_tree.json"
    gh = {str(node_id) for node_id in gh_ids}
    if not path.is_file():
        return [], {}, set(gh)
    tree = load_search_tree(path)
    sn_path, core = gold_path_core(tree, dag, gh)
    if not core:
        core = set(gh) & dag.node_id_set()
    return sibling_rank_pairs_on_path(tree, sn_path), claim_search_features(tree), core


def keep_set_from_sibling_rank(
    tree: Mapping[str, Any],
    scores: Mapping[str, float],
) -> set[str]:
    """Keep the highest-NR last_claim_id in each sibling group, plus the root."""
    nodes = {
        str(node["id"]): node
        for node in (tree.get("nodes") or [])
        if node.get("id")
    }
    kept: set[str] = set()
    children: dict[str, list[str]] = defaultdict(list)
    for node_id, node in nodes.items():
        parent_id = node.get("parent_id")
        claim_id = node.get("last_claim_id")
        if not parent_id:
            if claim_id:
                kept.add(str(claim_id))
            continue
        children[str(parent_id)].append(node_id)

    for kid_ids in children.values():
        ranked: list[tuple[float, str]] = []
        for kid_id in kid_ids:
            claim_id = nodes[kid_id].get("last_claim_id")
            if not claim_id:
                continue
            cid = str(claim_id)
            if cid not in scores:
                continue
            ranked.append((float(scores[cid]), cid))
        if not ranked:
            continue
        best = max(score for score, _cid in ranked)
        for score, cid in ranked:
            if score >= best - 1e-12:
                kept.add(cid)
    return kept


def aligned_labels(
    dump_dirs: Sequence[Path],
    gh_id_sets: Sequence[Iterable[str]],
) -> tuple[list[list[ClaimPair]], list[SearchFeats]]:
    pairs_out: list[list[ClaimPair]] = []
    feats_out: list[SearchFeats] = []
    for dump_dir, gh_ids in zip(dump_dirs, gh_id_sets):
        pairs, feats = optional_tree_labels(dump_dir, gh_ids)
        pairs_out.append(pairs)
        feats_out.append(feats)
    return pairs_out, feats_out
