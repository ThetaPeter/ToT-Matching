"""ToT feasibility prior: NR scores on GA nodes, not GI and not proofs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

import numpy as np

from tot.ir.schema import ClaimDAG
from tot.route.flow_dagnn import EXTRA_FEATURES, KEEP_THRESHOLD


def embed_nodes(dag: ClaimDAG, embedder: object) -> dict[str, np.ndarray]:
    """Embed GA nodes with the same text format used at train and eval."""
    if not dag.nodes:
        return {}
    texts = [
        f"{node.id}: {node.statement} {node.natural_language}" for node in dag.nodes
    ]
    mat = embedder.embed(texts)  # type: ignore[attr-defined]
    return {
        node.id: np.asarray(mat[i], dtype=np.float64)
        for i, node in enumerate(dag.nodes)
    }


def embedder_for_route_weights(
    in_dim: int,
    *,
    persisted_kind: str = "",
    allow_download: bool = False,
):
    """Train/serve lock: pick hashed vs MiniLM from model in_dim."""
    from tot.align.embed import FALLBACK_DIM, MINILM_DIM, SentenceEmbedder

    expected_emb = int(in_dim) - EXTRA_FEATURES
    if expected_emb == FALLBACK_DIM:
        return SentenceEmbedder(allow_download=False, force_hashed=True)
    if expected_emb == MINILM_DIM:
        return SentenceEmbedder(allow_download=allow_download)
    kind = str(persisted_kind or "")
    raise SystemExit(
        f"route-weights in_dim={in_dim} is incompatible with hashed "
        f"({FALLBACK_DIM}+{EXTRA_FEATURES}) or MiniLM ({MINILM_DIM}+{EXTRA_FEATURES}) extras "
        f"embedders (persisted embedder={kind!r})."
    )


ScoreFn = Callable[[ClaimDAG], dict[str, float]]


class Scorer(Protocol):
    def score_nodes(
        self, dag: ClaimDAG, embeddings: Mapping[str, np.ndarray]
    ) -> dict[str, float]: ...


def feasibility_prior(
    model: Scorer,
    dag: ClaimDAG,
    embeddings: Mapping[str, np.ndarray],
) -> dict[str, float]:
    """Score every GA node. A controller can use this as the ToT feasibility prior."""
    return model.score_nodes(dag, embeddings)


def score_nodes_fn(model: Scorer, embeddings: Mapping[str, np.ndarray]) -> ScoreFn:
    """Callable ``score_nodes(dag) -> dict[id, float]`` bound to one embedding map."""

    def score_nodes(dag: ClaimDAG) -> dict[str, float]:
        return model.score_nodes(dag, embeddings)

    return score_nodes


def keep_set_from_scores(
    scores: Mapping[str, float],
    threshold: float = KEEP_THRESHOLD,
) -> set[str]:
    return {node_id for node_id, score in scores.items() if score >= threshold}
