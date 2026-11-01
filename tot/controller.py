"""ToT search tree (separate JSON) driving incremental claim-DAG updates."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tot.config import TOT_BEAM, TOT_DEPTH, TOT_K
from tot.converter.builder import apply_expansion, convert_writeup
from tot.ir.schema import ClaimDAG
from tot.ir.validate import is_theorem_solution
from tot.providers.base import Extractor, Reasoner

NodePrior = Callable[[ClaimDAG], dict[str, float]]


@dataclass
class SearchNode:
    """Informal proof state together with the last claim."""

    id: str
    parent_id: str | None
    proof_state: str
    last_claim: str
    last_claim_id: str | None
    sink_claim_id: str | None
    depth: int
    score: float
    branch_id: str
    children: list[str] = field(default_factory=list)


def run_tot(
    reasoner: Reasoner,
    extractor: Extractor,
    theorem: str,
    *,
    problem_id: str,
    k: int = TOT_K,
    depth: int = TOT_DEPTH,
    beam: int = TOT_BEAM,
    seed_dag: ClaimDAG | None = None,
    node_prior: NodePrior | None = None,
) -> tuple[ClaimDAG, dict[str, Any]]:
    """Expand the search tree; Converter updates the merged claim DAG.

    If ``node_prior`` is set, each child's score is ``reasoner.score`` plus the
    prior on the added claim id (simple sum). Omit it for the Phase-1 path.
    """
    dag = seed_dag or convert_writeup(
        extractor,
        theorem,
        "We restate the theorem as the target claim to prove.",
        problem_id=problem_id,
    )
    seed_tc = _first_tc(dag)
    nodes: dict[str, SearchNode] = {}
    root = SearchNode(
        id="sn_0",
        parent_id=None,
        proof_state=theorem,
        last_claim="",
        last_claim_id=seed_tc,
        sink_claim_id=_first_ts(dag),
        depth=0,
        score=1.0,
        branch_id="b0",
    )
    nodes[root.id] = root
    sn_i = 1
    br_i = 1
    frontier = [root]
    for _level in range(1, max(0, depth) + 1):
        scored: list[SearchNode] = []
        for parent in frontier:
            claims = list(reasoner.expand(parent.proof_state, parent.last_claim, k))
            for claim in claims:
                branch_id = f"b{br_i}"
                br_i += 1
                before = dag.node_id_set()
                dag = apply_expansion(extractor, dag, parent, claim, branch_id)
                added_id, sink_id = _added_and_sink(dag, before, claim)
                child = SearchNode(
                    id=f"sn_{sn_i}",
                    parent_id=parent.id,
                    proof_state=_next_state(parent.proof_state, claim),
                    last_claim=claim,
                    last_claim_id=added_id or parent.last_claim_id,
                    sink_claim_id=sink_id or added_id or parent.sink_claim_id,
                    depth=parent.depth + 1,
                    score=_score_with_prior(
                        reasoner, parent.proof_state, claim, dag, added_id, node_prior
                    ),
                    branch_id=branch_id,
                )
                sn_i += 1
                nodes[child.id] = child
                parent.children.append(child.id)
                scored.append(child)
        if not scored:
            break
        frontier = _sibling_rank_frontier(scored, beam)
    tree = _tree_dict(problem_id, nodes, dag)
    return dag, tree


def resume_tot(
    reasoner: Reasoner,
    extractor: Extractor,
    dag: ClaimDAG,
    tree: dict[str, Any],
    new_proof_state: str,
    *,
    extra_depth: int = 1,
    k: int = TOT_K,
    beam: int = TOT_BEAM,
    node_prior: NodePrior | None = None,
) -> tuple[ClaimDAG, dict[str, Any]]:
    """Resume ToT from a Lean-derived informal proof state. Does not rewrite GI.

    ``node_prior`` uses the same simple-sum combination as ``run_tot``.
    """
    nodes = _nodes_from_tree(tree)
    sn_i = _next_sn_index(nodes)
    br_i = _next_branch_index(nodes)
    parent = _best_leaf(nodes)
    resume = SearchNode(
        id=f"sn_{sn_i}",
        parent_id=parent.id if parent is not None else None,
        proof_state=new_proof_state,
        last_claim=parent.last_claim if parent is not None else "",
        last_claim_id=parent.last_claim_id if parent is not None else _first_tc(dag),
        sink_claim_id=parent.sink_claim_id if parent is not None else _first_ts(dag),
        depth=(parent.depth + 1) if parent is not None else 0,
        score=0.5,
        branch_id=f"b_resume_{br_i}",
    )
    sn_i += 1
    br_i += 1
    nodes[resume.id] = resume
    if parent is not None:
        parent.children.append(resume.id)
    frontier = [resume]
    for _level in range(max(1, extra_depth)):
        scored: list[SearchNode] = []
        for node in frontier:
            claims = list(reasoner.expand(node.proof_state, node.last_claim, k))
            for claim in claims:
                branch_id = f"b{br_i}"
                br_i += 1
                before = dag.node_id_set()
                dag = apply_expansion(extractor, dag, node, claim, branch_id)
                added_id, sink_id = _added_and_sink(dag, before, claim)
                child = SearchNode(
                    id=f"sn_{sn_i}",
                    parent_id=node.id,
                    proof_state=_next_state(node.proof_state, claim),
                    last_claim=claim,
                    last_claim_id=added_id or node.last_claim_id,
                    sink_claim_id=sink_id or added_id or node.sink_claim_id,
                    depth=node.depth + 1,
                    score=_score_with_prior(
                        reasoner, node.proof_state, claim, dag, added_id, node_prior
                    ),
                    branch_id=branch_id,
                )
                sn_i += 1
                nodes[child.id] = child
                node.children.append(child.id)
                scored.append(child)
        if not scored:
            break
        frontier = _sibling_rank_frontier(scored, beam)
    problem_id = str(tree.get("problem_id") or dag.problem_id)
    return dag, _tree_dict(problem_id, nodes, dag)


def _sibling_winners(children: list[SearchNode]) -> list[SearchNode]:
    """Keep the max-score child of each parent. Ties are all kept."""
    by_parent: dict[str | None, list[SearchNode]] = {}
    for child in children:
        by_parent.setdefault(child.parent_id, []).append(child)
    winners: list[SearchNode] = []
    for group in by_parent.values():
        best = max(float(child.score) for child in group)
        winners.extend(
            child for child in group if float(child.score) >= best - 1e-12
        )
    return winners


def _sibling_rank_frontier(children: list[SearchNode], beam: int) -> list[SearchNode]:
    """Sibling winners first, then the usual beam cut."""
    winners = _sibling_winners(children)
    winners.sort(key=lambda node: node.score, reverse=True)
    return winners[: max(1, beam)]


def _score_with_prior(
    reasoner: Reasoner,
    proof_state: str,
    claim: str,
    dag: ClaimDAG,
    added_claim_id: str | None,
    node_prior: NodePrior | None,
) -> float:
    """``base + prior_s``; ``prior_s`` is 0 when the prior is omitted or no added id."""
    base = float(reasoner.score(proof_state, claim))
    if node_prior is None or not added_claim_id:
        return base
    prior_s = float(node_prior(dag).get(added_claim_id, 0.0))
    return base + prior_s


def dump_search_tree(tree: dict[str, Any], path: str | Path | None = None) -> str:
    raw = json.dumps(tree, indent=2)
    if path is not None:
        Path(path).write_text(raw, encoding="utf-8")
    return raw


def _tree_dict(
    problem_id: str,
    nodes: dict[str, SearchNode],
    dag: ClaimDAG,
) -> dict[str, Any]:
    branches = _branches_from(nodes, dag)
    return {
        "problem_id": problem_id,
        "nodes": [
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "proof_state": node.proof_state,
                "last_claim": node.last_claim,
                "last_claim_id": node.last_claim_id,
                "sink_claim_id": node.sink_claim_id,
                "depth": node.depth,
                "score": node.score,
                "branch_id": node.branch_id,
            }
            for node in nodes.values()
        ],
        "branches": branches,
    }


def _branches_from(nodes: dict[str, SearchNode], dag: ClaimDAG) -> list[dict[str, str]]:
    leaves = [node for node in nodes.values() if not node.children]
    branches: list[dict[str, str]] = []
    for node in leaves:
        sink = node.sink_claim_id or node.last_claim_id
        if not sink:
            continue
        branches.append({"branch_id": node.branch_id, "sink_claim_id": sink})
    if not branches:
        fallback = _first_ts(dag) or (dag.nodes[-1].id if dag.nodes else "")
        if fallback:
            branches.append({"branch_id": "b0", "sink_claim_id": fallback})
    return branches


def _added_and_sink(
    dag: ClaimDAG,
    before: set[str],
    claim: str,
) -> tuple[str | None, str | None]:
    added = [node.id for node in dag.nodes if node.id not in before]
    claim_l = (claim or "").lower()
    added_id: str | None = None
    for node_id in added:
        if not is_theorem_solution(node_id):
            added_id = node_id
            break
    if added_id is None and added:
        added_id = added[0]
    if added_id is None:
        by_stmt = [
            node.id
            for node in dag.nodes
            if (node.statement or "") == claim or (node.natural_language or "") == claim
        ]
        added_id = by_stmt[-1] if by_stmt else None
    dead = "dead" in claim_l or "unused" in claim_l
    if dead:
        sink = next((node_id for node_id in added if node_id.startswith("l_dead")), added_id)
        return added_id, sink
    ts = _first_ts(dag)
    return added_id, ts or added_id


def _next_state(proof_state: str, claim: str) -> str:
    proof_state = proof_state.rstrip()
    if not proof_state:
        return claim
    return f"{proof_state}\n{claim}"


def _first_tc(dag: ClaimDAG) -> str | None:
    for node in dag.nodes:
        if node.id.startswith("tc_"):
            return node.id
    return dag.nodes[0].id if dag.nodes else None


def _first_ts(dag: ClaimDAG) -> str | None:
    for node in dag.nodes:
        if is_theorem_solution(node.id) and not node.id.startswith("ts_dead"):
            return node.id
    for node in dag.nodes:
        if is_theorem_solution(node.id):
            return node.id
    return None


def _nodes_from_tree(tree: dict[str, Any]) -> dict[str, SearchNode]:
    nodes: dict[str, SearchNode] = {}
    for raw in tree.get("nodes", []):
        node = SearchNode(
            id=str(raw["id"]),
            parent_id=raw.get("parent_id"),
            proof_state=str(raw.get("proof_state") or ""),
            last_claim=str(raw.get("last_claim") or ""),
            last_claim_id=raw.get("last_claim_id"),
            sink_claim_id=raw.get("sink_claim_id"),
            depth=int(raw.get("depth") or 0),
            score=float(raw.get("score") or 0.0),
            branch_id=str(raw.get("branch_id") or "b0"),
        )
        nodes[node.id] = node
    for node in nodes.values():
        if node.parent_id and node.parent_id in nodes:
            parent = nodes[node.parent_id]
            if node.id not in parent.children:
                parent.children.append(node.id)
    return nodes


def _next_sn_index(nodes: dict[str, SearchNode]) -> int:
    best = 0
    for node_id in nodes:
        if node_id.startswith("sn_"):
            suffix = node_id[3:]
            if suffix.isdigit():
                best = max(best, int(suffix) + 1)
    return best


def _next_branch_index(nodes: dict[str, SearchNode]) -> int:
    best = 0
    for node in nodes.values():
        bid = node.branch_id
        if bid.startswith("b") and bid[1:].isdigit():
            best = max(best, int(bid[1:]) + 1)
    return best


def _best_leaf(nodes: dict[str, SearchNode]) -> SearchNode | None:
    leaves = [node for node in nodes.values() if not node.children]
    if not leaves:
        return next(iter(nodes.values()), None)
    leaves.sort(key=lambda node: node.score, reverse=True)
    return leaves[0]
