from __future__ import annotations

import json

from tot.data.minif2f import PairedProblem
from tot.ir.dump import dumps, load
from tot.pipeline import Phase1Providers, run_phase1
from tot.prove.nl_proof import gi_payload_ids, serialize_gi
from tot.providers.base import MockEmbedder, MockExtractor, MockProver, MockReasoner


def _problem(pid: str, statement: str, proof: str) -> PairedProblem:
    return PairedProblem(id=pid, statement=statement, human_proof=proof)


SYN_A = _problem(
    "syn_a",
    "Prove that 1+1=2.",
    "By the definition of addition, one plus one equals two, which is the required identity.",
)
SYN_B = _problem(
    "syn_b",
    "Prove that the square of an even integer is even.",
    "Let n=2k. Then n squared is 4k squared, hence two times 2k squared, so the square is even.",
)


def _providers(*, fail_times: int = 0) -> Phase1Providers:
    return Phase1Providers(
        reasoner=MockReasoner(include_dead=False),
        extractor=MockExtractor(),
        embedder=MockEmbedder(),
        prover=MockProver(fail_times=fail_times),
    )


def test_pipeline_smoke_two_synthetic_problems() -> None:
    first = run_phase1(
        SYN_A,
        _providers(fail_times=1),
        k=2,
        depth=2,
        beam=2,
        lean_resumes=1,
    )
    f1 = first.metrics["alignment_f1"]
    assert isinstance(f1, float)
    assert 0.0 <= f1 <= 1.0
    assert first.lean.ok
    assert first.metrics["lean_success"] is True
    assert first.metrics["lean_resumes"] == 1
    assert first.gi_json_before_lean == first.gi_json_after_lean
    assert gi_payload_ids(serialize_gi(first.gi)) == first.gi.node_id_set()
    assert "branches" in first.search_tree
    assert first.search_tree_json
    assert '"salience"' in first.salience_json or first.gh.nodes

    second = run_phase1(
        SYN_B,
        _providers(fail_times=0),
        k=2,
        depth=1,
        beam=2,
        lean_resumes=1,
    )
    assert isinstance(second.metrics["alignment_f1"], float)
    assert 0.0 <= second.metrics["alignment_f1"] <= 1.0
    assert second.lean.ok


def test_lean_fail_then_success_does_not_rewrite_gi() -> None:
    result = run_phase1(
        SYN_A,
        _providers(fail_times=1),
        k=2,
        depth=1,
        beam=2,
        lean_resumes=1,
    )
    assert result.lean.ok
    gi_json = dumps(result.gi)
    assert result.gi_json_before_lean == result.gi_json_after_lean == gi_json
    payload = serialize_gi(result.gi)
    assert gi_payload_ids(payload) == result.gi.node_id_set()
    for node in payload["nodes"]:
        assert "search_node_id" not in node
        assert "branch_id" not in node
        assert "salience" not in node
        assert "support" not in node
        assert "natural_language" not in node


def test_phase1_dump_dir_writes_ga_and_gh(tmp_path) -> None:
    result = run_phase1(
        SYN_A,
        _providers(fail_times=0),
        k=2,
        depth=1,
        beam=2,
        lean_resumes=0,
        dump_dir=tmp_path,
    )
    ga = load(tmp_path / "ga.json")
    gh = load(tmp_path / "gh.json")
    assert ga.node_id_set() == result.ga.node_id_set()
    assert gh.node_id_set() == result.gh.node_id_set()
    dumped = {path.name for path in tmp_path.iterdir()}
    assert {"ga.json", "gh.json", "gi.json", "metrics.json"} <= dumped
    gi = load(tmp_path / "gi.json")
    assert gi.node_id_set() == result.gi.node_id_set()
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "alignment_f1" in metrics
    assert set(metrics["gi_ids"]) == result.gi.node_id_set()
