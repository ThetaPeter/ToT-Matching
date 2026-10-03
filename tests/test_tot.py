from __future__ import annotations

import json

from tot.determinator.support import SearchBranch, support_core
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.providers.base import MockExtractor, MockReasoner
from tot.controller import dump_search_tree, run_tot


def _node(node_id: str, deps: list[str] | None = None) -> ClaimNode:
    return ClaimNode(
        id=node_id,
        natural_language=node_id,
        statement=node_id,
        dependencies=list(deps or []),
    )


def test_tot_produces_branches() -> None:
    dag, tree = run_tot(
        MockReasoner(include_dead=True),
        MockExtractor(),
        "Prove 1+1=2.",
        problem_id="tot_branches",
        k=2,
        depth=2,
        beam=2,
    )
    branches = tree["branches"]
    assert branches
    assert all("branch_id" in branch and "sink_claim_id" in branch for branch in branches)
    sinks = {branch["sink_claim_id"] for branch in branches}
    assert sinks <= dag.node_id_set()
    assert dag.nodes


def test_search_tree_json_separate_from_claim_dag() -> None:
    dag, tree = run_tot(
        MockReasoner(include_dead=False),
        MockExtractor(),
        "Prove even plus even is even.",
        problem_id="tot_separate",
        k=2,
        depth=1,
        beam=2,
    )
    raw = dump_search_tree(tree)
    parsed = json.loads(raw)
    assert "branches" in parsed
    assert "nodes" in parsed
    assert parsed["problem_id"] == "tot_separate"
    tree_ids = {node["id"] for node in parsed["nodes"]}
    assert all(node_id.startswith("sn_") for node_id in tree_ids)
    assert dag.node_id_set().isdisjoint(tree_ids)
    assert "natural_language" not in parsed["nodes"][0]


def test_support_core_fixture_gi_ids_unchanged() -> None:
    ga = ClaimDAG(
        problem_id="support_fixture",
        nodes=[
            _node("tc_1", []),
            _node("l1", ["tc_1"]),
            _node("ts_1", ["l1"]),
            _node("l_dead", ["tc_1"]),
        ],
    )
    branches = [SearchBranch(branch_id=f"ok_{i}", sink_claim_id="ts_1") for i in range(9)]
    branches.append(SearchBranch(branch_id="fail_0", sink_claim_id="l_dead"))
    gi = support_core(ga, branches, t=0.90)
    assert {node.id for node in gi.nodes} == {"tc_1", "l1", "ts_1"}
