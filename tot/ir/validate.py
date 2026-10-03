"""DAG checks for the dumpable claim IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tot.ir.schema import ClaimDAG

IssueCode = Literal["duplicate_id", "missing_dependency", "cycle", "no_outgoing"]


@dataclass(frozen=True)
class ValidationIssue:
    code: IssueCode
    message: str
    node_ids: tuple[str, ...]


class ClaimDAGValidationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        summary = "; ".join(issue.message for issue in issues) or "invalid claim DAG"
        super().__init__(summary)


def is_theorem_solution(node_id: str) -> bool:
    return node_id.startswith("ts_")


def validate(dag: ClaimDAG) -> list[ValidationIssue]:
    """Return structured issues. Empty list means the DAG is valid."""
    issues: list[ValidationIssue] = []
    ids = [node.id for node in dag.nodes]
    seen: set[str] = set()
    duplicates: list[str] = []
    for node_id in ids:
        if node_id in seen and node_id not in duplicates:
            duplicates.append(node_id)
        seen.add(node_id)
    if duplicates:
        issues.append(
            ValidationIssue(
                code="duplicate_id",
                message=f"duplicate node ids: {duplicates}",
                node_ids=tuple(duplicates),
            )
        )

    id_set = set(ids)
    missing_pairs: list[tuple[str, str]] = []
    for node in dag.nodes:
        for dep in node.dependencies:
            if dep not in id_set:
                missing_pairs.append((node.id, dep))
    if missing_pairs:
        involved = tuple(dict.fromkeys(dep for _, dep in missing_pairs))
        detail = ", ".join(f"{src}→{dep}" for src, dep in missing_pairs)
        issues.append(
            ValidationIssue(
                code="missing_dependency",
                message=f"missing dependency ids: {detail}",
                node_ids=involved,
            )
        )

    issues.extend(_cycle_issues(dag, id_set))
    issues.extend(_outgoing_issues(dag))
    return issues


def validate_or_raise(dag: ClaimDAG) -> None:
    issues = validate(dag)
    if issues:
        raise ClaimDAGValidationError(issues)


def _cycle_issues(dag: ClaimDAG, id_set: set[str]) -> list[ValidationIssue]:
    # Kahn on existing nodes only; missing deps are reported separately.
    indegree: dict[str, int] = {node.id: 0 for node in dag.nodes}
    successors: dict[str, list[str]] = {node.id: [] for node in dag.nodes}
    for node in dag.nodes:
        for dep in node.dependencies:
            if dep not in id_set:
                continue
            indegree[node.id] += 1
            successors[dep].append(node.id)

    queue = [node_id for node_id, deg in indegree.items() if deg == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if visited == len(dag.nodes):
        return []

    cyclic = tuple(sorted(node_id for node_id, deg in indegree.items() if deg > 0))
    return [
        ValidationIssue(
            code="cycle",
            message=f"cyclic dependencies involving {list(cyclic)}",
            node_ids=cyclic,
        )
    ]


def _outgoing_issues(dag: ClaimDAG) -> list[ValidationIssue]:
    referenced = {dep for node in dag.nodes for dep in node.dependencies}
    orphans = [
        node.id
        for node in dag.nodes
        if not is_theorem_solution(node.id) and node.id not in referenced
    ]
    if not orphans:
        return []
    return [
        ValidationIssue(
            code="no_outgoing",
            message=(
                "non-ts nodes must appear in some other node's dependencies: "
                f"{orphans}"
            ),
            node_ids=tuple(orphans),
        )
    ]
