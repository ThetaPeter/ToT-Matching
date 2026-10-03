"""JSON dump/load for the claim DAG. Dumpable JSON is the source of truth."""

from __future__ import annotations

from pathlib import Path

from tot.ir.schema import ClaimDAG


def dumps(dag: ClaimDAG) -> str:
    return dag.to_json()


def loads(data: str | bytes) -> ClaimDAG:
    return ClaimDAG.from_json(data)


def dump(dag: ClaimDAG, path: str | Path) -> None:
    Path(path).write_text(dumps(dag), encoding="utf-8")


def load(path: str | Path) -> ClaimDAG:
    return loads(Path(path).read_text(encoding="utf-8"))
