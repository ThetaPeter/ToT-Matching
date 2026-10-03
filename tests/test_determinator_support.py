"""Determinator support-core: search branches vote via sink ancestors (including self)."""

from __future__ import annotations

import pytest

from tot.config import SUPPORT_THRESHOLD
from tot.determinator.support import (
    SearchBranch,
    ancestors_including_self,
    compute_support,
    support_core,
)
from tot.ir.schema import ClaimDAG, ClaimNode


def _node(node_id: str, deps: list[str] | None = None) -> ClaimNode:
    return ClaimNode(
        id=node_id,
        natural_language=node_id,
        statement=node_id,
        dependencies=list(deps or []),
    )


def _ga() -> ClaimDAG:
    """Shared backbone tc_1 → l1 → ts_1 plus failed side claim l_dead."""
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
    """9 successful branches sink at ts_1; 1 failed branch sinks at l_dead."""
    successful = [
        SearchBranch(branch_id=f"ok_{i}", sink_claim_id="ts_1") for i in range(9)
    ]
    failed = [SearchBranch(branch_id="fail_0", sink_claim_id="l_dead")]
    return successful + failed


def test_ancestor_includes_self() -> None:
    dag = _ga()
    assert "ts_1" in ancestors_including_self(dag, "ts_1")
    assert ancestors_including_self(dag, "ts_1") == {"tc_1", "l1", "ts_1"}
    assert ancestors_including_self(dag, "l_dead") == {"tc_1", "l_dead"}


def test_support_t_090_gi_drops_failed_side_claim() -> None:
    dag = _ga()
    branches = _branches()
    supports = compute_support(dag, branches)
    assert supports["tc_1"] == pytest.approx(1.0)
    assert supports["l1"] == pytest.approx(0.9)
    assert supports["ts_1"] == pytest.approx(0.9)
    assert supports["l_dead"] == pytest.approx(0.1)

    assert SUPPORT_THRESHOLD == 0.90
    gi = support_core(dag, branches, t=0.90)
    assert {node.id for node in gi.nodes} == {"tc_1", "l1", "ts_1"}
    assert {node.id for node in gi.nodes} == dag.node_id_set() - {"l_dead"}
    assert gi.edge_set() == {("tc_1", "l1"), ("l1", "ts_1")}
    written = dag.node_map()
    assert written["tc_1"].support == pytest.approx(1.0)
    assert written["l_dead"].support == pytest.approx(0.1)


def test_support_t_095_keeps_only_tc_1() -> None:
    """ts_1 support is 0.9 because 9/10 sinks have ts_1 as ancestor-or-self.

    The failed branch sinks at l_dead, so it does not vote for ts_1 or l1.
    tc_1 is an ancestor of every sink, so it stays at t=0.95.
    """
    dag = _ga()
    gi = support_core(dag, _branches(), t=0.95)
    assert {node.id for node in gi.nodes} == {"tc_1"}
    assert gi.edge_set() == set()
    by_id = dag.node_map()
    assert by_id["ts_1"].support == pytest.approx(0.9)
    assert by_id["l1"].support == pytest.approx(0.9)
    assert by_id["l1"].support < 0.95
    assert by_id["ts_1"].support < 0.95
