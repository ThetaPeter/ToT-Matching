"""Phase-1 prompted pipeline: writeup → ToT → GH/GI → GI-only proof and Lean.

Optional Phase-2 ``node_prior`` is a ToT feasibility prior (NR scores on GA).
It does not define GI and does not generate proofs or Lean.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tot.align.match import match_assignment, match_hitter
from tot.align.salience import dump_gh, embed_dag, score_salience
from tot.config import SUPPORT_THRESHOLD, TOT_BEAM, TOT_DEPTH, TOT_K
from tot.converter.builder import convert_writeup
from tot.data.minif2f import PairedProblem
from tot.determinator.support import support_core
from tot.eval.metrics import lean_success, node_set_f1, prune_ratio
from tot.ir.dump import dump, dumps
from tot.ir.schema import ClaimDAG
from tot.lean.prover import LeanResult, run_prover
from tot.prove.nl_proof import prove_nl, serialize_gi
from tot.providers.base import Embedder, Extractor, Prover, Reasoner
from tot.controller import dump_search_tree, resume_tot, run_tot

NodePrior = Callable[[ClaimDAG], dict[str, float]]


@dataclass
class Phase1Providers:
    reasoner: Reasoner
    extractor: Extractor
    embedder: Embedder
    prover: Prover


@dataclass
class Phase1Result:
    ga: ClaimDAG
    gaprime: ClaimDAG
    gh: ClaimDAG
    gi: ClaimDAG
    search_tree: dict[str, Any]
    nl_proof: str
    lean: LeanResult
    metrics: dict[str, Any]
    salience_json: str
    search_tree_json: str
    gi_json_before_lean: str
    gi_json_after_lean: str


def run_phase1(
    problem: PairedProblem,
    providers: Phase1Providers,
    *,
    k: int = TOT_K,
    depth: int = TOT_DEPTH,
    beam: int = TOT_BEAM,
    support_t: float = SUPPORT_THRESHOLD,
    lean_resumes: int = 1,
    node_prior: NodePrior | None = None,
    dump_dir: str | Path | None = None,
) -> Phase1Result:
    gaprime = convert_writeup(
        providers.extractor,
        problem.statement,
        problem.human_proof,
        problem_id=f"{problem.id}_human",
    )
    ga, tree = run_tot(
        providers.reasoner,
        providers.extractor,
        problem.statement,
        problem_id=problem.id,
        k=k,
        depth=depth,
        beam=beam,
        node_prior=node_prior,
    )
    gh, gi = _align_and_gi(
        ga, gaprime, tree, providers.embedder, support_t=support_t
    )
    nl = prove_nl(providers.reasoner, gi)
    gi_before = dumps(gi)
    payload_ids = {node["id"] for node in serialize_gi(gi)["nodes"]}
    if payload_ids != gi.node_id_set():
        raise RuntimeError("NL/Lean payload must be GI node ids only")
    lean = run_prover(providers.prover, gi)
    gi_after = dumps(gi)
    if gi_before != gi_after:
        raise RuntimeError("Lean adapter rewrote GI")

    resumes = 0
    while (not lean.ok) and resumes < lean_resumes:
        resumes += 1
        ga, tree = resume_tot(
            providers.reasoner,
            providers.extractor,
            ga,
            tree,
            lean.new_proof_state,
            extra_depth=1,
            k=k,
            beam=beam,
            node_prior=node_prior,
        )
        gh, gi = _align_and_gi(
            ga, gaprime, tree, providers.embedder, support_t=support_t
        )
        nl = prove_nl(providers.reasoner, gi)
        gi_before = dumps(gi)
        lean = run_prover(providers.prover, gi)
        gi_after = dumps(gi)
        if gi_before != gi_after:
            raise RuntimeError("Lean adapter rewrote GI")

    metrics = {
        "lean_success": lean_success(lean),
        "alignment_f1": node_set_f1(gi.node_id_set(), gh.node_id_set()),
        "prune_ratio": prune_ratio(gi, ga),
        "gi_ids": sorted(gi.node_id_set()),
        "gh_ids": sorted(gh.node_id_set()),
        "lean_resumes": resumes,
        "k": int(k),
        "depth": int(depth),
        "beam": int(beam),
        "t": float(support_t),
        "lean_resumes_budget": int(lean_resumes),
    }
    if dump_dir is not None:
        out = Path(dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        dump(ga, out / "ga.json")
        dump(gh, out / "gh.json")
        dump(gi, out / "gi.json")
        (out / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        (out / "nl_proof.txt").write_text(nl, encoding="utf-8")
        (out / "lean.lean").write_text(lean.code or "", encoding="utf-8")
        (out / "search_tree.json").write_text(dump_search_tree(tree), encoding="utf-8")
    return Phase1Result(
        ga=ga,
        gaprime=gaprime,
        gh=gh,
        gi=gi,
        search_tree=tree,
        nl_proof=nl,
        lean=lean,
        metrics=metrics,
        salience_json=dump_gh(gh),
        search_tree_json=dump_search_tree(tree),
        gi_json_before_lean=gi_before,
        gi_json_after_lean=gi_after,
    )


def _align_and_gi(
    ga: ClaimDAG,
    gaprime: ClaimDAG,
    tree: dict[str, Any],
    embedder: Embedder,
    *,
    support_t: float,
) -> tuple[ClaimDAG, ClaimDAG]:
    emb_a = embed_dag(ga, embedder)
    emb_p = embed_dag(gaprime, embedder)
    mapping = match_assignment(ga, gaprime, emb_a, emb_p)
    gh = match_hitter(ga, gaprime, emb_a, emb_p)
    score_salience(gh, gaprime, embedder, mapping)
    branches = list(tree.get("branches") or [])
    gi = support_core(ga, branches, t=support_t)
    return gh, gi
