"""Salience as a GH node feature (embedding / token-overlap), not a second network."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from tot.align.match import cosine_similarity
from tot.ir.dump import dump, dumps
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.providers.base import Embedder


def node_text(node: ClaimNode) -> str:
    return (node.natural_language or node.statement or node.id).strip()


def score_salience(
    gh: ClaimDAG,
    gaprime: ClaimDAG,
    embedder: Embedder,
    mapping: Mapping[str, str] | None = None,
) -> ClaimDAG:
    """Write ``salience`` onto GH nodes vs GA′ texts; return the same GH object."""
    if not gh.nodes:
        return gh
    texts_h = [node_text(node) for node in gh.nodes]
    vecs_h = np.asarray(embedder.embed(texts_h), dtype=float)
    if not gaprime.nodes:
        for node in gh.nodes:
            node.salience = 0.0
        return gh
    texts_p = [node_text(node) for node in gaprime.nodes]
    vecs_p = np.asarray(embedder.embed(texts_p), dtype=float)
    gp_index = {node.id: i for i, node in enumerate(gaprime.nodes)}
    mapping = dict(mapping or {})
    for i, node in enumerate(gh.nodes):
        matched = mapping.get(node.id)
        if matched is not None and matched in gp_index:
            node.salience = _unit_cosine(vecs_h[i], vecs_p[gp_index[matched]])
            continue
        best = 0.0
        for j in range(len(gaprime.nodes)):
            best = max(best, _unit_cosine(vecs_h[i], vecs_p[j]))
        node.salience = best
    return gh


def dump_gh(gh: ClaimDAG, path: str | Path | None = None) -> str:
    raw = dumps(gh)
    if path is not None:
        dump(gh, path)
    return raw


def embed_dag(dag: ClaimDAG, embedder: Embedder) -> dict[str, np.ndarray]:
    if not dag.nodes:
        return {}
    texts = [node_text(node) for node in dag.nodes]
    vecs = np.asarray(embedder.embed(texts), dtype=float)
    return {node.id: vecs[i] for i, node in enumerate(dag.nodes)}


def _unit_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, cosine_similarity(a, b))
