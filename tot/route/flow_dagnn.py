"""FlowDAGNN: two-pass DAG GNN with Kirchhoff-style flow attention.

Edges are ``u → v`` when ``u`` is required for ``v`` (``v`` lists ``u`` in
``dependencies``). Outgoing neighbors of ``u`` are successors: nodes that
depend on ``u``. Flow attention softmax-normalizes scores across those
outgoing neighbors (resource conservation), not incoming neighbors (GAT).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from tot.ir.schema import ClaimDAG, ClaimNode
from tot.route._autograd import Tensor, as_tensor, bce_with_logits, sigmoid, softmax, tanh

KEEP_THRESHOLD = 0.5
EXTRA_FEATURES = 2


def pos_weight_from_graphs(graphs: Sequence[tuple[ClaimDAG, set[str]]]) -> float:
    """BCE pos-weight = n_drop / n_keep from GA nodes on this train set.

    Equals (1 - p) / p for keep-rate p. Returns 1.0 if there are no keep or
    no drop nodes (unweighted).
    """
    n_nodes = 0
    n_keep = 0
    for dag, keep in graphs:
        ids = dag.node_id_set()
        n_nodes += len(ids)
        n_keep += sum(1 for node_id in ids if node_id in keep)
    if n_keep <= 0 or n_keep >= n_nodes:
        return 1.0
    return float(n_nodes - n_keep) / float(n_keep)


def topological_ids(dag: ClaimDAG) -> list[str]:
    """Deterministic Kahn order; sources first."""
    id_set = dag.node_id_set()
    indegree = {node.id: 0 for node in dag.nodes}
    successors: dict[str, list[str]] = {node.id: [] for node in dag.nodes}
    for node in dag.nodes:
        for dep in node.dependencies:
            if dep not in id_set:
                raise KeyError(f"missing dependency {dep!r} of {node.id!r}")
            indegree[node.id] += 1
            successors[dep].append(node.id)
    for node_id in successors:
        successors[node_id].sort()
    queue = sorted(node_id for node_id, deg in indegree.items() if deg == 0)
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
                queue.sort()
    if len(order) != len(dag.nodes):
        raise ValueError("claim DAG has a cycle")
    return order


def successor_map(dag: ClaimDAG) -> dict[str, list[str]]:
    succ = {node.id: [] for node in dag.nodes}
    id_set = dag.node_id_set()
    for node in dag.nodes:
        for dep in node.dependencies:
            if dep not in id_set:
                raise KeyError(f"missing dependency {dep!r} of {node.id!r}")
            succ[dep].append(node.id)
    for node_id in succ:
        succ[node_id].sort()
    return succ


def predecessor_map(dag: ClaimDAG) -> dict[str, list[str]]:
    return {node.id: sorted(node.dependencies) for node in dag.nodes}


def node_feature(node: ClaimNode, embeddings: Mapping[str, np.ndarray]) -> np.ndarray:
    if node.id not in embeddings:
        raise KeyError(f"missing embedding for node {node.id!r}")
    emb = np.asarray(embeddings[node.id], dtype=np.float64).reshape(-1)
    extra = np.array(
        [
            0.0 if node.salience is None else float(node.salience),
            # Always 0: `support` is a post-search Determinator bit. A live ToT
            # prior scores a growing DAG and would not see it, so training must
            # not leak that extra. EXTRA_FEATURES stays 2 (in_dim = emb + 2).
            0.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate([emb, extra])


def build_features(dag: ClaimDAG, embeddings: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {node.id: node_feature(node, embeddings) for node in dag.nodes}


def _dot(left: Tensor, right: Tensor) -> Tensor:
    return (left * right).sum()


def _param(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> Tensor:
    return Tensor(rng.uniform(-scale, scale, size=shape))


class FlowDAGNN:
    """Numpy FlowDAGNN with a node-level keep/feasible head."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 8,
        n_layers: int = 1,
        seed: int = 0,
    ) -> None:
        if in_dim <= 0 or hidden_dim <= 0 or n_layers <= 0:
            raise ValueError("in_dim, hidden_dim, and n_layers must be positive")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        rng = np.random.default_rng(seed)
        in_scale = 1.0 / np.sqrt(self.in_dim)
        hid_scale = 1.0 / np.sqrt(self.hidden_dim)
        self._params: dict[str, Tensor] = {
            "W_in": _param(rng, (self.in_dim, self.hidden_dim), in_scale),
            "b_in": Tensor(np.zeros(self.hidden_dim)),
            "w_head_fw": _param(rng, (self.hidden_dim,), hid_scale),
            "w_head_rv": _param(rng, (self.hidden_dim,), hid_scale),
            "w_head_x": _param(rng, (self.in_dim,), in_scale),
            "b_head": Tensor(0.0),
        }
        for layer in range(self.n_layers):
            self._params.update(
                {
                    f"l{layer}_rv_w1": _param(rng, (self.hidden_dim,), hid_scale),
                    f"l{layer}_rv_w2": _param(rng, (self.hidden_dim,), hid_scale),
                    f"l{layer}_rv_Wh": _param(rng, (self.hidden_dim, self.hidden_dim), hid_scale),
                    f"l{layer}_rv_Wm": _param(rng, (self.hidden_dim, self.hidden_dim), hid_scale),
                    f"l{layer}_rv_b": Tensor(np.zeros(self.hidden_dim)),
                    f"l{layer}_fw_w1": _param(rng, (self.hidden_dim,), hid_scale),
                    f"l{layer}_fw_w2": _param(rng, (self.hidden_dim,), hid_scale),
                    f"l{layer}_fw_Wh": _param(rng, (self.hidden_dim, self.hidden_dim), hid_scale),
                    f"l{layer}_fw_Wm": _param(rng, (self.hidden_dim, self.hidden_dim), hid_scale),
                    f"l{layer}_fw_b": Tensor(np.zeros(self.hidden_dim)),
                }
            )
        self._last_flow: dict[str, dict[str, float]] = {}
        self._last_reverse: dict[str, dict[str, float]] = {}
        self._last_order: list[str] = []

    def named_parameters(self) -> dict[str, Tensor]:
        return dict(self._params)

    def parameters(self) -> list[Tensor]:
        return list(self._params.values())

    def step(self, lr: float, clip: float = 5.0) -> None:
        total = 0.0
        for param in self._params.values():
            total += float(np.sum(param.grad * param.grad))
        norm = float(np.sqrt(total))
        scale = float(lr)
        if clip is not None and norm > clip:
            scale *= clip / (norm + 1e-12)
        for param in self._params.values():
            param.data -= scale * param.grad

    def _update(self, hidden: Tensor, message: Tensor, w_h: Tensor, w_m: Tensor, bias: Tensor) -> Tensor:
        return tanh(hidden @ w_h + message @ w_m + bias) + hidden

    def _forward(
        self,
        dag: ClaimDAG,
        embeddings: Mapping[str, np.ndarray],
    ) -> dict[str, Tensor]:
        features = build_features(dag, embeddings)
        for node_id, feat in features.items():
            if feat.shape[0] != self.in_dim:
                raise ValueError(
                    f"feature dim {feat.shape[0]} for {node_id!r} != model in_dim {self.in_dim}"
                )
        topo = topological_ids(dag)
        succ = successor_map(dag)
        pred = predecessor_map(dag)
        zeros = Tensor(np.zeros(self.hidden_dim))
        x_t = {node_id: Tensor(feat) for node_id, feat in features.items()}
        hidden = {node_id: tanh(x_t[node_id] @ self._params["W_in"] + self._params["b_in"]) for node_id in topo}
        last_flow: dict[str, dict[str, float]] = {node_id: {} for node_id in topo}
        last_reverse: dict[str, dict[str, float]] = {node_id: {} for node_id in topo}

        for layer in range(self.n_layers):
            w1_rv = self._params[f"l{layer}_rv_w1"]
            w2_rv = self._params[f"l{layer}_rv_w2"]
            rv_wh = self._params[f"l{layer}_rv_Wh"]
            rv_wm = self._params[f"l{layer}_rv_Wm"]
            rv_b = self._params[f"l{layer}_rv_b"]
            w1_fw = self._params[f"l{layer}_fw_w1"]
            w2_fw = self._params[f"l{layer}_fw_w2"]
            fw_wh = self._params[f"l{layer}_fw_Wh"]
            fw_wm = self._params[f"l{layer}_fw_Wm"]
            fw_b = self._params[f"l{layer}_fw_b"]

            h_in = hidden
            h_rv: dict[str, Tensor] = {}
            for node_id in reversed(topo):
                outgoing = succ[node_id]
                if not outgoing:
                    message = zeros
                else:
                    scores = [_dot(w1_rv, h_in[node_id]) + _dot(w2_rv, h_rv[dst]) for dst in outgoing]
                    weights = softmax(scores)
                    message = weights[0] * h_rv[outgoing[0]]
                    for weight, dst in zip(weights[1:], outgoing[1:]):
                        message = message + weight * h_rv[dst]
                    last_reverse[node_id] = {
                        dst: float(weight.data) for dst, weight in zip(outgoing, weights)
                    }
                h_rv[node_id] = self._update(h_in[node_id], message, rv_wh, rv_wm, rv_b)

            h_fw: dict[str, Tensor] = {}
            flow_tensors: dict[str, dict[str, Tensor]] = {node_id: {} for node_id in topo}
            for node_id in topo:
                incoming = pred[node_id]
                message = zeros
                for src in incoming:
                    if not flow_tensors[src]:
                        outgoing = succ[src]
                        scores = [
                            _dot(w1_fw, h_rv[dst]) + _dot(w2_fw, h_fw[src]) for dst in outgoing
                        ]
                        weights = softmax(scores)
                        flow_tensors[src] = {
                            dst: weight for dst, weight in zip(outgoing, weights)
                        }
                        last_flow[src] = {
                            dst: float(weight.data) for dst, weight in zip(outgoing, weights)
                        }
                    message = message + flow_tensors[src][node_id] * h_fw[src]
                h_fw[node_id] = self._update(h_rv[node_id], message, fw_wh, fw_wm, fw_b)
            hidden = h_fw
            self._last_reverse = last_reverse
            self._last_flow = last_flow

        self._last_order = list(topo)
        w_fw = self._params["w_head_fw"]
        w_rv = self._params["w_head_rv"]
        w_x = self._params["w_head_x"]
        b_head = self._params["b_head"]
        logits: dict[str, Tensor] = {}
        for node_id in topo:
            logits[node_id] = (
                _dot(w_fw, hidden[node_id])
                + _dot(w_rv, h_rv[node_id])
                + _dot(w_x, x_t[node_id])
                + b_head
            )
        return logits

    def logits(self, dag: ClaimDAG, embeddings: Mapping[str, np.ndarray]) -> dict[str, float]:
        return {node_id: float(logit.data) for node_id, logit in self._forward(dag, embeddings).items()}

    def score_nodes(self, dag: ClaimDAG, embeddings: Mapping[str, np.ndarray]) -> dict[str, float]:
        """Feasibility scores in (0, 1) for every GA node id."""
        scored: dict[str, float] = {}
        for node_id, logit in self._forward(dag, embeddings).items():
            scored[node_id] = float(sigmoid(as_tensor(float(logit.data))).data)
        return scored

    def predict_keep_set(
        self,
        dag: ClaimDAG,
        embeddings: Mapping[str, np.ndarray],
        threshold: float = KEEP_THRESHOLD,
    ) -> set[str]:
        scores = self.score_nodes(dag, embeddings)
        return {node_id for node_id, score in scores.items() if score >= threshold}

    def outgoing_attention(
        self,
        dag: ClaimDAG,
        embeddings: Mapping[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Forward-pass flow weights: sender → {successor: β}. Rows sum to 1."""
        self._forward(dag, embeddings)
        return {src: dict(dsts) for src, dsts in self._last_flow.items()}

    def reverse_attention(
        self,
        dag: ClaimDAG,
        embeddings: Mapping[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """Reverse-pass weights over original outgoing neighbors."""
        self._forward(dag, embeddings)
        return {src: dict(dsts) for src, dsts in self._last_reverse.items()}

    def flow_attention_matrix(
        self,
        dag: ClaimDAG,
        embeddings: Mapping[str, np.ndarray],
    ) -> tuple[list[str], np.ndarray]:
        """Dense A[i, j] = flow from ``ids[i]`` to ``ids[j]``. Rows = sources."""
        ids = [node.id for node in dag.nodes]
        attn = self.outgoing_attention(dag, embeddings)
        matrix = np.zeros((len(ids), len(ids)), dtype=np.float64)
        pos = {node_id: i for i, node_id in enumerate(ids)}
        for src, dsts in attn.items():
            for dst, weight in dsts.items():
                matrix[pos[src], pos[dst]] = weight
        return ids, matrix

    def node_loss(
        self,
        dag: ClaimDAG,
        keep: set[str],
        embeddings: Mapping[str, np.ndarray],
        pos_weight: float = 1.0,
    ) -> Tensor:
        logits = self._forward(dag, embeddings)
        weight = float(pos_weight)
        losses: list[Tensor] = []
        for node in dag.nodes:
            target = 1.0 if node.id in keep else 0.0
            loss = bce_with_logits(logits[node.id], target)
            if target == 1.0 and weight != 1.0:
                loss = loss * weight
            losses.append(loss)
        total = losses[0]
        for loss in losses[1:]:
            total = total + loss
        return total / float(len(losses))

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "in_dim": np.array(self.in_dim),
            "hidden_dim": np.array(self.hidden_dim),
            "n_layers": np.array(self.n_layers),
        }
        for name, param in self._params.items():
            payload[name] = np.array(param.data, copy=True)
        return payload

    def load_state_dict(self, payload: Mapping[str, np.ndarray]) -> None:
        for name, param in self._params.items():
            if name not in payload:
                raise KeyError(f"missing parameter {name!r}")
            param.data = np.array(payload[name], dtype=np.float64, copy=True)


def keep_ids_from_label(label: set[str] | ClaimDAG) -> set[str]:
    if isinstance(label, ClaimDAG):
        return label.node_id_set()
    return set(label)
