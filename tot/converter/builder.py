"""Incremental ProofFlow-style claim DAG conversion with repair + validate."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tot.ir.dump import dumps
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.ir.validate import (
    ClaimDAGValidationError,
    is_theorem_solution,
    validate,
    validate_or_raise,
)
from tot.providers.base import Extractor

ALLOWED_NODE_KEYS = {
    "id",
    "natural_language",
    "statement",
    "dependencies",
    "search_node_id",
    "branch_id",
    "salience",
    "support",
}

MAX_REPAIR = 4


def convert_writeup(
    extractor: Extractor,
    theorem: str,
    proof: str,
    *,
    problem_id: str = "writeup",
) -> ClaimDAG:
    """Human (or seed) writeup → GA′ / seed GA. Repair loop then validate."""
    context: dict[str, Any] = {"problem_id": problem_id}
    raw = extractor.extract(theorem, proof, context=context)
    dag = parse_extractor_json(raw, problem_id)
    return _ensure_valid(extractor, dag, context, theorem=theorem, proof=proof)


def apply_expansion(
    extractor: Extractor,
    dag: ClaimDAG,
    search_node: Any,
    new_claim_text: str,
    branch_id: str,
) -> ClaimDAG:
    """Merge a ToT expansion into the claim DAG and tag new nodes."""
    sn_id = _search_attr(search_node, "id")
    proof_state = str(_search_attr(search_node, "proof_state") or "")
    last_claim = str(_search_attr(search_node, "last_claim") or "")
    last_claim_id = _search_attr(search_node, "last_claim_id")
    context: dict[str, Any] = {
        "problem_id": dag.problem_id,
        "existing": json.loads(dumps(dag)),
        "search_node_id": sn_id,
        "last_claim": last_claim,
        "last_claim_id": last_claim_id,
        "branch_id": branch_id,
        "new_claim_text": new_claim_text,
    }
    original_ids = dag.node_id_set()
    raw = extractor.extract(proof_state, new_claim_text, context=context)
    incoming = parse_extractor_json(raw, dag.problem_id)
    merged = merge_incremental(
        dag,
        incoming,
        search_node_id=sn_id,
        branch_id=branch_id,
        original_ids=original_ids,
    )
    return _ensure_valid(
        extractor,
        merged,
        context,
        theorem=proof_state,
        proof=new_claim_text,
    )


def merge_incremental(
    base: ClaimDAG,
    incoming: ClaimDAG,
    *,
    search_node_id: str | int | None,
    branch_id: str | None,
    original_ids: set[str],
) -> ClaimDAG:
    out = {node.id: node.model_copy() for node in base.nodes}
    for node in incoming.nodes:
        if node.id not in out:
            out[node.id] = node.model_copy(
                update={
                    "search_node_id": search_node_id,
                    "branch_id": branch_id,
                }
            )
            continue
        if is_theorem_solution(node.id) or node.id not in original_ids:
            update: dict[str, Any] = {"dependencies": list(node.dependencies)}
            if node.id not in original_ids:
                update["search_node_id"] = search_node_id
                update["branch_id"] = branch_id
            if node.natural_language:
                update["natural_language"] = node.natural_language
            if node.statement:
                update["statement"] = node.statement
            out[node.id] = out[node.id].model_copy(update=update)
    return ClaimDAG(problem_id=base.problem_id, nodes=list(out.values()))


def parse_extractor_json(raw: str | Mapping[str, Any], problem_id: str) -> ClaimDAG:
    if isinstance(raw, Mapping):
        data: Any = dict(raw)
    else:
        data = json.loads(raw)
    if isinstance(data, list):
        data = {"problem_id": problem_id, "nodes": data}
    if not isinstance(data, dict):
        raise ClaimDAGValidationError([])
    data.setdefault("problem_id", problem_id)
    nodes_in = data.get("nodes", [])
    cleaned: list[dict[str, Any]] = []
    if isinstance(nodes_in, dict):
        items = list(nodes_in.items())
        for key, node in items:
            payload = dict(node) if isinstance(node, Mapping) else {"id": str(key)}
            payload.setdefault("id", str(key))
            cleaned.append(_coerce_node_payload(payload, str(key)))
    else:
        for i, node in enumerate(nodes_in or []):
            if isinstance(node, ClaimNode):
                cleaned.append(json.loads(node.model_dump_json()))
                continue
            payload = dict(node) if isinstance(node, Mapping) else {}
            cleaned.append(_coerce_node_payload(payload, f"l{i + 1}"))
    data["nodes"] = cleaned
    return ClaimDAG.model_validate(data)


def _coerce_node_payload(payload: Mapping[str, Any], fallback_id: str) -> dict[str, Any]:
    raw = {k: v for k, v in payload.items() if k in ALLOWED_NODE_KEYS}
    raw.setdefault("id", fallback_id)
    statement = raw.get("statement") or raw.get("natural_language") or raw["id"]
    natural = raw.get("natural_language") or statement
    raw["statement"] = str(statement)
    raw["natural_language"] = str(natural)
    raw.setdefault("dependencies", [])
    return raw


def fix_common_issues(dag: ClaimDAG) -> ClaimDAG:
    """Deterministic structural repair used by the extractor repair loop / last resort."""
    seen: set[str] = set()
    nodes: list[ClaimNode] = []
    for node in dag.nodes:
        if node.id in seen:
            continue
        seen.add(node.id)
        nodes.append(node.model_copy())
    id_set = {node.id for node in nodes}
    for node in nodes:
        node.dependencies = [dep for dep in node.dependencies if dep in id_set]

    dag2 = ClaimDAG(problem_id=dag.problem_id, nodes=nodes)
    issues = validate(dag2)
    guard = 0
    while any(issue.code == "cycle" for issue in issues) and guard < len(dag2.nodes) + 2:
        cyclic = next(issue for issue in issues if issue.code == "cycle")
        victim = cyclic.node_ids[0] if cyclic.node_ids else dag2.nodes[0].id
        for node in dag2.nodes:
            if node.id == victim and node.dependencies:
                node.dependencies = list(node.dependencies[:-1])
                break
        issues = validate(dag2)
        guard += 1

    issues = validate(dag2)
    orphans: list[str] = []
    for issue in issues:
        if issue.code == "no_outgoing":
            orphans.extend(issue.node_ids)
    if orphans:
        ts_nodes = [node for node in dag2.nodes if is_theorem_solution(node.id)]
        if not ts_nodes:
            dag2.nodes.append(
                ClaimNode(
                    id="ts_1",
                    natural_language="theorem",
                    statement="theorem",
                    dependencies=list(dict.fromkeys(orphans)),
                )
            )
        else:
            target = ts_nodes[0]
            deps = list(target.dependencies)
            for orphan in orphans:
                if orphan not in deps:
                    deps.append(orphan)
            target.dependencies = deps
    return dag2


def _ensure_valid(
    extractor: Extractor,
    dag: ClaimDAG,
    context: Mapping[str, Any],
    *,
    theorem: str,
    proof: str,
) -> ClaimDAG:
    ctx = dict(context)
    ctx.setdefault("theorem", theorem)
    ctx.setdefault("proof", proof)
    for _ in range(MAX_REPAIR):
        issues = validate(dag)
        if not issues:
            validate_or_raise(dag)
            return dag
        raw = extractor.repair(dumps(dag), issues, context=ctx)
        dag = parse_extractor_json(raw, dag.problem_id)
    dag = fix_common_issues(dag)
    validate_or_raise(dag)
    return dag


def _search_attr(search_node: Any, name: str) -> Any:
    if isinstance(search_node, Mapping):
        return search_node.get(name)
    return getattr(search_node, name, None)
