from __future__ import annotations

from pathlib import Path

from tot.data.minif2f import is_full_human_proof, load_paired, main

TINY_ROOT = Path(__file__).parent / "fixtures" / "minif2f_tiny"
KEEP_IDS = {"tiny_even", "tiny_sum"}
DISCARD_IDS = {"tiny_stub"}


def test_tiny_keep_discard_id_sets() -> None:
    kept, discarded = load_paired(TINY_ROOT)
    assert {problem.id for problem in kept} == KEEP_IDS
    assert set(discarded) == DISCARD_IDS
    for problem in kept:
        assert is_full_human_proof(problem.human_proof)


def test_is_full_human_proof_rejects_stub() -> None:
    assert not is_full_human_proof("")
    assert not is_full_human_proof("TODO")
    assert not is_full_human_proof("sorry")
    assert is_full_human_proof(
        "Let n = 2k. Then n^2 = 4k^2 = 2(2k^2), so the square is even."
    )


def test_cli_prints_paired_n_and_discarded_count(capsys) -> None:
    code = main(["--root", str(TINY_ROOT)])
    assert code == 0
    out = capsys.readouterr().out
    assert "paired N=2" in out
    assert "discarded count=1" in out
