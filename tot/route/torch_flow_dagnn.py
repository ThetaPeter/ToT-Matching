"""PyTorch FlowDAGNN: outgoing Kirchhoff-style flow attention (no GAT).

Public scoring contract matches the numpy ``FlowDAGNN`` used by ToT priors:
``score_nodes(dag, embeddings) -> dict[id, float]`` in ``[0, 1]``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tot.ir.schema import ClaimDAG
from tot.route.flow_dagnn import (
    KEEP_THRESHOLD,
    EXTRA_FEATURES,
    _path_rank_terms,
    build_features,
    pos_weight_from_graphs,
    predecessor_map,
    successor_map,
    topological_ids,
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for tot.route.torch_flow_dagnn. "
        "Install with: pip install torch"
    ) from exc


class TorchFlowDAGNN(nn.Module):
    """Two-pass DAG GNN with softmax over *outgoing* successors only."""

    KEEP_THRESHOLD = KEEP_THRESHOLD

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        n_layers: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or n_layers <= 0:
            raise ValueError("in_dim, hidden_dim, and n_layers must be positive")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        torch.manual_seed(seed)
        self.input = nn.Linear(self.in_dim, self.hidden_dim)
        self.fw_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layers)]
        )
        self.bw_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_layers)]
        )
        # Attention: score = a^T tanh(W_src h_u + W_dst h_v) on edge u→v (outgoing from u)
        self.attn_src = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_dst = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.attn_vec = nn.Linear(self.hidden_dim, 1, bias=False)
        self.head = nn.Linear(self.hidden_dim * 2 + self.in_dim, 1)
        self._last_outgoing: dict[str, dict[str, float]] = {}

    def _edge_score(self, h_src: torch.Tensor, h_dst: torch.Tensor) -> torch.Tensor:
        return self.attn_vec(torch.tanh(self.attn_src(h_src) + self.attn_dst(h_dst))).squeeze(-1)

    def forward_logits(
        self,
        dag: ClaimDAG,
        embeddings: Mapping[str, np.ndarray],
        search_feats: Mapping[str, tuple[float, float]] | None = None,
    ) -> dict[str, torch.Tensor]:
        feats = build_features(dag, embeddings, search_feats=search_feats)
        order = topological_ids(dag)
        succ = successor_map(dag)
        pred = predecessor_map(dag)
        x = {
            node_id: torch.as_tensor(feats[node_id], dtype=torch.float32)
            for node_id in order
        }
        for node_id, tensor in x.items():
            if tensor.numel() != self.in_dim:
                raise ValueError(
                    f"feature dim {tensor.numel()} for {node_id!r} != in_dim {self.in_dim}"
                )
        h = {node_id: torch.tanh(self.input(x[node_id])) for node_id in order}
        outgoing_attn: dict[str, dict[str, float]] = {node_id: {} for node_id in order}

        for layer in range(self.n_layers):
            # Backward pass along reverse topo (messages from successors via outgoing attn)
            h_bw: dict[str, torch.Tensor] = {}
            for node_id in reversed(order):
                outs = succ[node_id]
                if not outs:
                    msg = torch.zeros(self.hidden_dim)
                else:
                    scores = torch.stack(
                        [self._edge_score(h[node_id], h[dst]) for dst in outs]
                    )
                    weights = torch.softmax(scores, dim=0)
                    msg = sum(weights[i] * h[outs[i]] for i in range(len(outs)))
                    outgoing_attn[node_id] = {
                        outs[i]: float(weights[i].detach()) for i in range(len(outs))
                    }
                h_bw[node_id] = torch.tanh(self.bw_layers[layer](h[node_id] + msg))

            # Forward pass: gather from predecessors using their outgoing weights
            h_fw: dict[str, torch.Tensor] = {}
            cached_weights: dict[str, dict[str, torch.Tensor]] = {}
            for node_id in order:
                parents = pred[node_id]
                msg = torch.zeros(self.hidden_dim)
                for src in parents:
                    if src not in cached_weights:
                        outs = succ[src]
                        scores = torch.stack(
                            [self._edge_score(h_bw[src], h_bw[dst]) for dst in outs]
                        )
                        weights = torch.softmax(scores, dim=0)
                        cached_weights[src] = {
                            outs[i]: weights[i] for i in range(len(outs))
                        }
                        outgoing_attn[src] = {
                            outs[i]: float(weights[i].detach()) for i in range(len(outs))
                        }
                    msg = msg + cached_weights[src][node_id] * h_fw.get(src, h_bw[src])
                h_fw[node_id] = torch.tanh(self.fw_layers[layer](h_bw[node_id] + msg))
            h = h_fw

        self._last_outgoing = outgoing_attn
        logits: dict[str, torch.Tensor] = {}
        for node_id in order:
            logits[node_id] = self.head(
                torch.cat([h[node_id], h_bw[node_id], x[node_id]], dim=0)
            ).squeeze(-1)
        return logits

    def score_nodes(
        self,
        dag: ClaimDAG,
        embeddings: Mapping[str, np.ndarray],
        search_feats: Mapping[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward_logits(dag, embeddings, search_feats=search_feats)
            return {
                node_id: float(torch.sigmoid(logit))
                for node_id, logit in logits.items()
            }

    def outgoing_attention(
        self,
        dag: ClaimDAG,
        embeddings: Mapping[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        self.score_nodes(dag, embeddings)
        return {key: dict(val) for key, val in self._last_outgoing.items()}

    def node_loss(
        self,
        dag: ClaimDAG,
        keep_ids: set[str],
        embeddings: Mapping[str, np.ndarray],
        pos_weight: float = 1.0,
        pairs: Sequence[tuple[str, str]] | None = None,
        search_feats: Mapping[str, tuple[float, float]] | None = None,
        path_rank: tuple[Sequence[str], Sequence[Sequence[str]]] | None = None,
    ) -> torch.Tensor:
        logits = self.forward_logits(dag, embeddings, search_feats=search_feats)
        weight = float(pos_weight)
        losses = []
        for node_id, logit in logits.items():
            target = torch.tensor(
                1.0 if node_id in keep_ids else 0.0, dtype=torch.float32
            )
            loss = nn.functional.binary_cross_entropy_with_logits(logit, target)
            if float(target) == 1.0 and weight != 1.0:
                loss = loss * weight
            losses.append(loss)
        bce = torch.stack(losses).mean()
        rank_pairs = [
            (pos, neg)
            for pos, neg in (pairs or [])
            if pos in logits and neg in logits and pos != neg
        ]
        extra = bce
        if rank_pairs:
            rank_terms = []
            for pos, neg in rank_pairs:
                score_pos = torch.sigmoid(logits[pos])
                score_neg = torch.sigmoid(logits[neg])
                rank_terms.append(nn.functional.softplus(score_neg - score_pos))
            extra = extra + torch.stack(rank_terms).mean()
        path_terms = _path_rank_terms(
            logits, path_rank, torch.sigmoid, nn.functional.softplus
        )
        if path_terms:
            extra = extra + torch.stack(path_terms).mean()
        return extra

    def state_dict_numpy(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": np.array("torch"),
            "in_dim": np.array(self.in_dim),
            "hidden_dim": np.array(self.hidden_dim),
            "n_layers": np.array(self.n_layers),
        }
        for key, value in self.state_dict().items():
            payload[f"param::{key}"] = value.detach().cpu().numpy()
        return payload

    def load_state_dict_numpy(self, payload: Mapping[str, np.ndarray]) -> None:
        tensors = {
            key.removeprefix("param::"): torch.as_tensor(value)
            for key, value in payload.items()
            if str(key).startswith("param::")
        }
        self.load_state_dict(tensors, strict=True)


def train_torch_route_network(
    graphs: list[tuple[ClaimDAG, set[str]]],
    embeddings: Mapping[str, np.ndarray] | list[Mapping[str, np.ndarray]],
    *,
    epochs: int = 200,
    lr: float = 0.05,
    hidden_dim: int = 32,
    n_layers: int = 2,
    seed: int = 0,
    pos_weight: float | None = None,
    sibling_pairs: Sequence[Sequence[tuple[str, str]]] | None = None,
    search_feats: Sequence[Mapping[str, tuple[float, float]]] | None = None,
    path_ranks: Sequence[tuple[Sequence[str], Sequence[Sequence[str]]] | None]
    | None = None,
) -> TorchFlowDAGNN:
    """SGD on node-wise pos-weighted BCE plus sibling ranking."""
    first_dag, _ = graphs[0]
    if isinstance(embeddings, list):
        first_emb = embeddings[0]
    else:
        first_emb = embeddings
    first_search = search_feats[0] if search_feats else None
    feats = build_features(first_dag, first_emb, search_feats=first_search)
    in_dim = next(iter(feats.values())).shape[0]
    if in_dim < EXTRA_FEATURES:
        raise ValueError("node features must include embedding dims plus extras")
    model = TorchFlowDAGNN(
        in_dim=in_dim, hidden_dim=hidden_dim, n_layers=n_layers, seed=seed
    )
    pw = float(pos_weight) if pos_weight is not None else pos_weight_from_graphs(graphs)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(int(epochs)):
        optim.zero_grad()
        losses = []
        for i, (dag, keep) in enumerate(graphs):
            emb = embeddings[i] if isinstance(embeddings, list) else embeddings
            pairs = sibling_pairs[i] if sibling_pairs is not None else None
            feats_i = search_feats[i] if search_feats is not None else None
            path_i = path_ranks[i] if path_ranks is not None else None
            losses.append(
                model.node_loss(
                    dag,
                    keep,
                    emb,
                    pos_weight=pw,
                    pairs=pairs,
                    search_feats=feats_i,
                    path_rank=path_i,
                )
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        optim.step()
    return model
