from __future__ import annotations

from tot.providers.base import MockExtractor, MockReasoner
from tot.controller import resume_tot, run_tot

# MockExtractor writeup uses l1; first expansion id is l3 (see _expansion_json).
BOOST_CLAIM_ID = "l3"


def _full_id_prior(boost_id: str, boost: float = 10.0):
    calls: list[set[str]] = []

    def node_prior(dag):
        ids = dag.node_id_set()
        scores = {node_id: 0.0 for node_id in ids}
        assert set(scores) == ids
        calls.append(set(scores))
        if boost_id in scores:
            scores[boost_id] = boost
        return scores

    return node_prior, calls


def test_run_tot_without_prior_still_returns_branches() -> None:
    dag, tree = run_tot(
        MockReasoner(include_dead=False),
        MockExtractor(),
        "Prove 1+1=2.",
        problem_id="tot_no_prior",
        k=2,
        depth=1,
        beam=2,
    )
    assert tree["branches"]
    assert dag.nodes


def test_prior_boosts_known_added_claim() -> None:
    prior, calls = _full_id_prior(BOOST_CLAIM_ID, 10.0)
    dag, tree = run_tot(
        MockReasoner(include_dead=False),
        MockExtractor(),
        "Prove 1+1=2.",
        problem_id="tot_prior_boost",
        k=2,
        depth=1,
        beam=2,
        node_prior=prior,
    )
    assert calls
    assert calls[-1] == dag.node_id_set()
    children = [node for node in tree["nodes"] if node["depth"] == 1]
    assert children
    best = max(children, key=lambda node: node["score"])
    assert best["last_claim_id"] == BOOST_CLAIM_ID
    assert best["score"] > 10.0
    beam_ids = {node["last_claim_id"] for node in children}
    assert BOOST_CLAIM_ID in beam_ids


def test_resume_tot_accepts_node_prior() -> None:
    extractor = MockExtractor()
    dag, tree = run_tot(
        MockReasoner(include_dead=False),
        extractor,
        "Prove even plus even is even.",
        problem_id="tot_resume_prior",
        k=2,
        depth=1,
        beam=2,
    )
    prior, calls = _full_id_prior("", 0.0)
    dag, tree = resume_tot(
        MockReasoner(include_dead=False),
        extractor,
        dag,
        tree,
        "Resume from a Lean failure with a new informal state.",
        extra_depth=1,
        k=2,
        beam=2,
        node_prior=prior,
    )
    assert tree["branches"]
    assert calls
    assert calls[-1] == dag.node_id_set()
