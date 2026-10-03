from __future__ import annotations

from tot.converter.builder import apply_expansion, convert_writeup
from tot.ir.dump import dumps
from tot.ir.validate import validate
from tot.providers.base import MockExtractor


def test_convert_writeup_valid_proofflow_ids() -> None:
    extractor = MockExtractor()
    dag = convert_writeup(
        extractor,
        "Prove 1+1=2.",
        "By definition of addition, 1+1 equals 2 as required.",
        problem_id="writeup_ok",
    )
    assert validate(dag) == []
    ids = dag.node_id_set()
    assert "tc_1" in ids
    assert "l1" in ids
    assert "ts_1" in ids
    assert ids == {"tc_1", "l1", "ts_1"}


def test_repair_invalid_then_valid() -> None:
    extractor = MockExtractor(invalid_first=True)
    dag = convert_writeup(
        extractor,
        "Prove n even implies n^2 even.",
        "Write n=2k so n^2=4k^2=2(2k^2) and conclude evenness.",
        problem_id="repair_me",
    )
    assert validate(dag) == []
    assert "ghost_lemma" not in {dep for node in dag.nodes for dep in node.dependencies}
    assert "ts_1" in dag.node_id_set()


def test_apply_expansion_merge_tags_search_and_branch() -> None:
    extractor = MockExtractor()
    dag = convert_writeup(
        extractor,
        "Prove 2+2=4.",
        "Two plus two is four by addition of natural numbers.",
        problem_id="merge",
    )
    before = dag.node_id_set()
    search_node = {
        "id": "sn_7",
        "proof_state": "Prove 2+2=4.",
        "last_claim": "",
        "last_claim_id": "l1",
    }
    dag = apply_expansion(
        extractor,
        dag,
        search_node,
        "Advance the argument: add the remaining successor step.",
        "b_fork",
    )
    assert validate(dag) == []
    added = dag.node_id_set() - before
    assert added
    tagged = [node for node in dag.nodes if node.id in added]
    assert tagged
    assert all(node.search_node_id == "sn_7" for node in tagged)
    assert all(node.branch_id == "b_fork" for node in tagged)
    dumps(dag)
