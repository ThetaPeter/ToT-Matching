from __future__ import annotations

from tot.determinator.support import SearchBranch, support_core
from tot.ir.dump import dumps
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.lean.prover import run_prover
from tot.prove.nl_proof import build_gi_payload, gi_payload_ids, prove_nl, serialize_gi
from tot.providers.base import MockProver, MockReasoner


def _node(node_id: str, deps: list[str] | None = None) -> ClaimNode:
    return ClaimNode(
        id=node_id,
        natural_language=node_id,
        statement=node_id,
        dependencies=list(deps or []),
    )


def _ga_with_extra() -> ClaimDAG:
    return ClaimDAG(
        problem_id="support_fixture",
        nodes=[
            _node("tc_1", []),
            _node("l1", ["tc_1"]),
            _node("ts_1", ["l1"]),
            _node("l_dead", ["tc_1"]),
        ],
    )


def _branches() -> list[SearchBranch]:
    successful = [
        SearchBranch(branch_id=f"ok_{i}", sink_claim_id="ts_1") for i in range(9)
    ]
    failed = [SearchBranch(branch_id="fail_0", sink_claim_id="l_dead")]
    return successful + failed


GI_IDS = {"tc_1", "l1", "ts_1"}


def test_proof_payload_from_ga_is_gi_ids_only() -> None:
    ga = _ga_with_extra()
    payload = build_gi_payload(ga, _branches(), t=0.90)
    assert gi_payload_ids(payload) == GI_IDS
    assert "l_dead" not in gi_payload_ids(payload)
    dumped = dumps(ga)
    assert "l_dead" in dumped


def test_nl_and_lean_payloads_gi_ids_only() -> None:
    ga = _ga_with_extra()
    gi = support_core(ga, _branches(), t=0.90)
    assert gi.node_id_set() == GI_IDS
    proof_payload = serialize_gi(gi)
    assert gi_payload_ids(proof_payload) == GI_IDS
    for node in proof_payload["nodes"]:
        assert set(node.keys()) == {"id", "statement", "dependencies"}
    reasoner = MockReasoner()
    prove_nl(reasoner, gi)
    assert reasoner.last_prove_payload is not None
    assert gi_payload_ids(reasoner.last_prove_payload) == GI_IDS

    prover = MockProver()
    before = dumps(gi)
    result = run_prover(prover, gi)
    assert dumps(gi) == before
    assert prover.last_payload is not None
    assert gi_payload_ids(prover.last_payload) == GI_IDS
    assert "l_dead" not in gi_payload_ids(prover.last_payload)
    assert result.ok
