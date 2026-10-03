from __future__ import annotations

from tot.ir.dump import dump, load, loads
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.ir.validate import ClaimDAGValidationError, validate, validate_or_raise


def _node(node_id: str, deps: list[str] | None = None, **fields: object) -> ClaimNode:
    payload = {
        "id": node_id,
        "natural_language": fields.pop("natural_language", node_id),
        "statement": fields.pop("statement", node_id),
        "dependencies": list(deps or []),
    }
    payload.update(fields)
    return ClaimNode.model_validate(payload)


def _proofflow_dag() -> ClaimDAG:
    return ClaimDAG(
        problem_id="proofflow_ok",
        nodes=[
            _node("tc_1", []),
            _node("def_1", []),
            _node("l1", ["tc_1", "def_1"]),
            _node("l2", ["l1"]),
            _node("ts_1", ["l2"]),
        ],
    )


def test_acyclic_proofflow_like_graph_passes() -> None:
    dag = _proofflow_dag()
    assert validate(dag) == []
    validate_or_raise(dag)


def test_cycle_fails_validation() -> None:
    dag = ClaimDAG(
        problem_id="cycle",
        nodes=[
            _node("l1", ["l2"]),
            _node("l2", ["l1"]),
        ],
    )
    issues = validate(dag)
    codes = {issue.code for issue in issues}
    assert "cycle" in codes
    cyclic_ids = set()
    for issue in issues:
        if issue.code == "cycle":
            cyclic_ids.update(issue.node_ids)
    assert cyclic_ids == {"l1", "l2"}
    try:
        validate_or_raise(dag)
        raise AssertionError("expected ClaimDAGValidationError")
    except ClaimDAGValidationError as exc:
        assert any(issue.code == "cycle" for issue in exc.issues)


def test_non_ts_without_outgoing_fails() -> None:
    dag = ClaimDAG(
        problem_id="orphan",
        nodes=[
            _node("tc_1", []),
            _node("l1", ["tc_1"]),
            _node("ts_1", ["l1"]),
            _node("def_orphan", []),
        ],
    )
    issues = validate(dag)
    outgoing = [issue for issue in issues if issue.code == "no_outgoing"]
    assert outgoing
    assert "def_orphan" in outgoing[0].node_ids
    assert "ts_1" not in outgoing[0].node_ids


def test_ts_sink_without_outgoing_is_ok() -> None:
    dag = ClaimDAG(
        problem_id="ts_sink",
        nodes=[
            _node("tc_1", []),
            _node("ts_1", ["tc_1"]),
        ],
    )
    assert validate(dag) == []
    assert "ts_1" not in {dep for node in dag.nodes for dep in node.dependencies}


def test_missing_dependency_fails() -> None:
    dag = ClaimDAG(
        problem_id="missing",
        nodes=[
            _node("tc_1", []),
            _node("l1", ["ghost_lemma"]),
            _node("ts_1", ["l1"]),
        ],
    )
    issues = validate(dag)
    missing = [issue for issue in issues if issue.code == "missing_dependency"]
    assert missing
    assert "ghost_lemma" in missing[0].node_ids
    try:
        validate_or_raise(dag)
        raise AssertionError("expected ClaimDAGValidationError")
    except ClaimDAGValidationError as exc:
        assert any(issue.code == "missing_dependency" for issue in exc.issues)


def test_round_trip_preserves_node_ids_and_edges(tmp_path) -> None:
    original = ClaimDAG(
        problem_id="example",
        nodes=[
            _node(
                "tc_1",
                [],
                natural_language="assume n is even",
                statement="Even n",
                search_node_id=3,
                branch_id="b0",
                salience=0.4,
                support=None,
            ),
            _node("l1", ["tc_1"], search_node_id="sn_l1"),
            _node("ts_1", ["l1"]),
        ],
    )
    loaded = loads(original.to_json())
    assert loaded.node_id_set() == original.node_id_set() == {"tc_1", "l1", "ts_1"}
    assert loaded.edge_set() == original.edge_set() == {("tc_1", "l1"), ("l1", "ts_1")}
    assert loaded.problem_id == "example"
    by_id = loaded.node_map()
    assert by_id["tc_1"].search_node_id == 3
    assert by_id["tc_1"].branch_id == "b0"
    assert by_id["tc_1"].salience == 0.4
    assert by_id["tc_1"].support is None
    assert by_id["l1"].search_node_id == "sn_l1"

    path = tmp_path / "dag.json"
    dump(original, path)
    from_file = load(path)
    assert from_file.node_id_set() == original.node_id_set()
    assert from_file.edge_set() == original.edge_set()


def test_nodes_dict_form_is_accepted() -> None:
    dag = ClaimDAG.model_validate(
        {
            "problem_id": "dict_nodes",
            "nodes": {
                "tc_1": {
                    "natural_language": "tc",
                    "statement": "tc",
                    "dependencies": [],
                },
                "ts_1": {
                    "id": "ts_1",
                    "natural_language": "ts",
                    "statement": "ts",
                    "dependencies": ["tc_1"],
                },
            },
        }
    )
    assert dag.node_id_set() == {"tc_1", "ts_1"}
    validate_or_raise(dag)
