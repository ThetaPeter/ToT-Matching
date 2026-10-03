"""Train FlowDAGNN on dumped GA/GH JSON (or in-memory claim DAGs)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from tot.ir.dump import load
from tot.ir.schema import ClaimDAG
from tot.route.flow_dagnn import (
    EXTRA_FEATURES,
    FlowDAGNN,
    build_features,
    keep_ids_from_label,
    pos_weight_from_graphs,
)


def load_labeled_pair(ga_path: str | Path, gh_path: str | Path) -> tuple[ClaimDAG, set[str]]:
    """Read one GA/GH JSON pair via dumpable IR load."""
    ga = load(ga_path)
    gh = load(gh_path)
    return ga, gh.node_id_set()


def _as_graphs(
    graphs: Sequence[tuple[ClaimDAG, set[str] | ClaimDAG]] | None,
    json_pairs: Sequence[tuple[str | Path, str | Path]] | None,
) -> list[tuple[ClaimDAG, set[str]]]:
    packed: list[tuple[ClaimDAG, set[str]]] = []
    if graphs:
        for dag, label in graphs:
            packed.append((dag, keep_ids_from_label(label)))
    if json_pairs:
        for ga_path, gh_path in json_pairs:
            packed.append(load_labeled_pair(ga_path, gh_path))
    if not packed:
        raise ValueError("train_route_network needs graphs or json_pairs")
    return packed


def _embed_map(
    dag: ClaimDAG,
    embeddings: Mapping[str, np.ndarray] | Sequence[Mapping[str, np.ndarray]],
    graph_i: int,
) -> Mapping[str, np.ndarray]:
    if isinstance(embeddings, Sequence) and not isinstance(embeddings, (str, bytes, Mapping)):
        return embeddings[graph_i]
    if dag.problem_id in embeddings:
        inner = embeddings[dag.problem_id]
        if isinstance(inner, Mapping) and all(node.id in inner for node in dag.nodes):
            sample = next(iter(inner.values()))
            if not isinstance(sample, Mapping):
                return inner
    return embeddings  # type: ignore[return-value]


def train_route_network(
    graphs: Sequence[tuple[ClaimDAG, set[str] | ClaimDAG]] | None = None,
    embeddings: Mapping[str, np.ndarray] | Sequence[Mapping[str, np.ndarray]] | None = None,
    epochs: int = 400,
    lr: float = 0.15,
    hidden_dim: int = 8,
    n_layers: int = 1,
    seed: int = 0,
    json_pairs: Sequence[tuple[str | Path, str | Path]] | None = None,
    backend: str = "numpy",
    pos_weight: float | None = None,
) -> Any:
    """SGD on node-wise pos-weighted BCE. Label 1 iff the GA node id is in GH.

    ``pos_weight`` defaults to n_drop/n_keep on this train set (not a constant).
    ``backend="torch"`` uses TorchFlowDAGNN (outgoing flow attention).
    ``backend="numpy"`` keeps the original autograd FlowDAGNN (pytest default).
    """
    if embeddings is None:
        raise ValueError("embeddings are required")
    labeled = _as_graphs(graphs, json_pairs)
    pw = float(pos_weight) if pos_weight is not None else pos_weight_from_graphs(labeled)
    if backend == "torch":
        from tot.route.torch_flow_dagnn import train_torch_route_network

        emb_list = [
            dict(_embed_map(dag, embeddings, i)) for i, (dag, _) in enumerate(labeled)
        ]
        return train_torch_route_network(
            [(dag, keep) for dag, keep in labeled],
            emb_list,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            seed=seed,
            pos_weight=pw,
        )
    if backend != "numpy":
        raise ValueError(f"unknown route backend {backend!r} (use numpy|torch)")
    first_dag, _ = labeled[0]
    first_emb = _embed_map(first_dag, embeddings, 0)
    feats = build_features(first_dag, first_emb)
    in_dim = next(iter(feats.values())).shape[0]
    if in_dim < EXTRA_FEATURES:
        raise ValueError(
            "node features must include embedding dims plus salience/support"
        )
    model = FlowDAGNN(in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers, seed=seed)
    for _ in range(int(epochs)):
        losses = []
        for i, (dag, keep) in enumerate(labeled):
            emb = _embed_map(dag, embeddings, i)
            losses.append(model.node_loss(dag, keep, emb, pos_weight=pw))
        loss = losses[0]
        for term in losses[1:]:
            loss = loss + term
        loss = loss / float(len(losses))
        loss.backward()
        model.step(lr)
    return model
