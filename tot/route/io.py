"""Dump/load FlowDAGNN weights as numpy ``.npz``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tot.route.flow_dagnn import FlowDAGNN


def route_meta_payload(
    *,
    keep_threshold: float,
    pos_weight: float | None,
    embedder: str,
) -> dict[str, np.ndarray]:
    """Run metadata stored alongside the state dict in an npz."""
    return {
        "keep_threshold": np.asarray(float(keep_threshold), dtype=np.float64),
        "pos_weight": (
            np.asarray(np.nan, dtype=np.float64)
            if pos_weight is None
            else np.asarray(float(pos_weight), dtype=np.float64)
        ),
        "embedder": np.asarray(str(embedder)),
    }


def dump_weights(
    model: FlowDAGNN,
    path: str | Path,
    *,
    keep_threshold: float,
    pos_weight: float | None,
    embedder: str,
) -> None:
    payload = model.state_dict()
    payload.update(
        route_meta_payload(
            keep_threshold=keep_threshold,
            pos_weight=pos_weight,
            embedder=embedder,
        )
    )
    np.savez(path, **payload)


def load_weights(path: str | Path) -> FlowDAGNN:
    payload = np.load(path, allow_pickle=False)
    model = FlowDAGNN(
        in_dim=int(payload["in_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        n_layers=int(payload["n_layers"]),
        seed=0,
    )
    arrays = {key: payload[key] for key in payload.files}
    model.load_state_dict(arrays)
    return model


def load_route_meta(path: str | Path) -> dict[str, Any]:
    """Return keep_threshold, pos_weight, embedder, in_dim from a weights npz."""
    payload = np.load(path, allow_pickle=False)
    files = set(payload.files)
    keep_threshold: float | None = None
    if "keep_threshold" in files:
        keep_threshold = float(np.asarray(payload["keep_threshold"]).reshape(()))
    pos_weight: float | None = None
    if "pos_weight" in files:
        raw_pw = float(np.asarray(payload["pos_weight"], dtype=np.float64).reshape(()))
        if np.isfinite(raw_pw):
            pos_weight = raw_pw
    embedder = ""
    if "embedder" in files:
        embedder = str(np.asarray(payload["embedder"]).reshape(()).item())
    return {
        "keep_threshold": keep_threshold,
        "pos_weight": pos_weight,
        "embedder": embedder,
        "in_dim": int(payload["in_dim"]),
    }


def load_route_model(path: str | Path, backend: str = "numpy"):
    """Load a numpy or torch FlowDAGNN scorer from an npz (extra keys ignored)."""
    path = Path(path)
    if backend == "numpy":
        return load_weights(path)
    if backend != "torch":
        raise SystemExit(f"Unknown route backend {backend!r} (use numpy|torch)")
    from tot.route.torch_flow_dagnn import TorchFlowDAGNN

    payload = np.load(path, allow_pickle=False)
    model = TorchFlowDAGNN(
        in_dim=int(payload["in_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        n_layers=int(payload["n_layers"]),
        seed=0,
    )
    model.load_state_dict_numpy({key: payload[key] for key in payload.files})
    return model
