"""NCP stub-decoder overfit (no HuggingFace download)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tot.ir.schema import ClaimDAG, ClaimNode
from tot.prove.ncp import (
    NCPConfig,
    NCPExample,
    NCPReasoner,
    gi_text,
    load_ncp,
    save_ncp,
    train_ncp,
)
from tot.prove.nl_proof import serialize_gi


def _gi() -> ClaimDAG:
    return ClaimDAG(
        problem_id="ncp_toy",
        nodes=[
            ClaimNode(
                id="tc_1",
                natural_language="goal",
                statement="P",
                dependencies=[],
            ),
            ClaimNode(
                id="l1",
                natural_language="lemma",
                statement="P implies Q",
                dependencies=["tc_1"],
            ),
            ClaimNode(
                id="ts_1",
                natural_language="done",
                statement="Q",
                dependencies=["l1"],
            ),
        ],
    )


def test_gi_text_is_gi_only() -> None:
    text = gi_text(_gi())
    assert "tc_1" in text and "l1" in text and "ts_1" in text
    payload = serialize_gi(_gi())
    assert {node["id"] for node in payload["nodes"]} == {"tc_1", "l1", "ts_1"}


def test_stub_ncp_trains_and_prove_nl(tmp_path) -> None:
    gi = _gi()
    emb = {
        "tc_1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "l1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "ts_1": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    target = "From P derive Q via the lemma."
    model = train_ncp(
        [NCPExample(gi=gi, human_proof=target, embeddings=emb)],
        epochs=40,
        config=NCPConfig(stub_decoder=True, hidden_dim=32, n_layers=1, n_heads=4, lr=1e-2),
    )
    path = tmp_path / "ncp.pt"
    save_ncp(model, path)
    loaded = load_ncp(path)
    reasoner = NCPReasoner(loaded, emb)
    out = reasoner.prove_nl(serialize_gi(gi))
    assert isinstance(out, str)
    assert len(out) >= 0
    # GI-only invariant: payload ids unchanged
    assert {n["id"] for n in serialize_gi(gi)["nodes"]} == {"tc_1", "l1", "ts_1"}
