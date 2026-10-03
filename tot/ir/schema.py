"""Dumpable claim-DAG IR. JSON is the source of truth."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClaimNode(BaseModel):
    """One claim in the merged graph.

    ProofFlow id prefixes: ``tc_*`` condition, ``def_*`` definition, ``l*`` lemma
    (e.g. ``l1``), ``ts_*`` theorem solution.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    natural_language: str
    statement: str
    dependencies: list[str] = Field(default_factory=list)
    search_node_id: str | int | None = None
    branch_id: str | None = None
    salience: float | None = None
    support: float | None = None


class ClaimDAG(BaseModel):
    """A dumpable claim DAG. Canonical JSON uses a node list."""

    model_config = ConfigDict(extra="forbid")

    problem_id: str
    nodes: list[ClaimNode] = Field(default_factory=list)

    @field_validator("nodes", mode="before")
    @classmethod
    def _coerce_nodes(cls, value: object) -> object:
        if isinstance(value, dict):
            coerced: list[Any] = []
            for key, node in value.items():
                if isinstance(node, ClaimNode):
                    coerced.append(node)
                    continue
                if isinstance(node, Mapping):
                    payload = dict(node)
                    payload.setdefault("id", str(key))
                    coerced.append(payload)
                    continue
                coerced.append(node)
            return coerced
        return value

    def node_map(self) -> dict[str, ClaimNode]:
        return {node.id: node for node in self.nodes}

    def node_id_set(self) -> set[str]:
        return {node.id for node in self.nodes}

    def edge_set(self) -> set[tuple[str, str]]:
        """Directed edges ``u → v``: ``u`` is required to prove ``v``."""
        edges: set[tuple[str, str]] = set()
        for node in self.nodes:
            for dep in node.dependencies:
                edges.add((dep, node.id))
        return edges

    def induced_subgraph(self, keep: Iterable[str]) -> ClaimDAG:
        kept = set(keep)
        nodes = [
            node.model_copy(
                update={
                    "dependencies": [dep for dep in node.dependencies if dep in kept],
                }
            )
            for node in self.nodes
            if node.id in kept
        ]
        return ClaimDAG(problem_id=self.problem_id, nodes=nodes)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, data: str | bytes) -> ClaimDAG:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return cls.model_validate_json(data)
