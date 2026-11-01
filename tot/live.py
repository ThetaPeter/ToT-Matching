"""Live / smoke runner for Phase-1 dumps (slice A).

Examples:
  python -m tot.live --provider mock --ids mathd_algebra_10 --dump-root data/dumps/valid
  python -m tot.live --provider live --split valid --limit 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from collections.abc import Callable

from tot.align.embed import SentenceEmbedder
from tot.config import SUPPORT_THRESHOLD, TOT_BEAM, TOT_DEPTH, TOT_K
from tot.data.minif2f import PairedProblem, load_paired
from tot.eval.metrics import lean_success, node_set_f1, prune_ratio
from tot.ir.schema import ClaimDAG
from tot.pipeline import Phase1Providers, run_phase1
from tot.providers.base import MockEmbedder, MockExtractor, MockProver, MockReasoner
from tot.route.flow_dagnn import EXTRA_FEATURES, KEEP_THRESHOLD
from tot.route.io import load_route_meta, load_route_model
from tot.route.prior import embed_nodes, embedder_for_route_weights, feasibility_prior, keep_set_from_scores
from tot.route.search_labels import keep_set_from_sibling_rank, load_search_tree

NodePrior = Callable[[ClaimDAG], dict[str, float]]

SMOKE_IDS = [
    "mathd_algebra_10",
    "mathd_algebra_11",
    "mathd_algebra_13",
    "mathd_algebra_15",
    "mathd_algebra_22",
    "mathd_algebra_28",
    "mathd_algebra_31",
    "mathd_algebra_35",
]


def _problems_for_split(
    root: Path,
    split: str,
) -> list[PairedProblem]:
    kept, _ = load_paired(root)
    by_id = {problem.id: problem for problem in kept}
    split_dir = root / "informal" / split
    if not split_dir.is_dir():
        # facebook layout sometimes uses test/valid at repo root informal/
        split_dir = root / "informal" / ("valid" if split == "valid" else "test")
    ids = sorted(path.stem for path in split_dir.glob("*.json"))
    return [by_id[i] for i in ids if i in by_id]


def _select_problems(
    root: Path,
    *,
    split: str,
    ids: list[str] | None,
    limit: int | None,
) -> list[PairedProblem]:
    if ids:
        kept, _ = load_paired(root)
        by_id = {problem.id: problem for problem in kept}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"Unknown or discarded problem ids: {missing}")
        return [by_id[i] for i in ids]
    problems = _problems_for_split(root, split)
    if limit is not None:
        problems = problems[: max(0, int(limit))]
    return problems


def resolve_run_knobs(
    *,
    skip_lean_compile: bool,
    lean_resumes: int | None,
    k: int | None,
    depth: int | None,
    beam: int | None,
    t: float | None,
) -> dict[str, int | float]:
    """CLI defaults stay thesis locks; skip-lean forces lean_resumes=0 unless set."""
    if lean_resumes is not None:
        resumes = int(lean_resumes)
    elif skip_lean_compile:
        resumes = 0
    else:
        resumes = 1
    return {
        "k": int(k) if k is not None else TOT_K,
        "depth": int(depth) if depth is not None else TOT_DEPTH,
        "beam": int(beam) if beam is not None else TOT_BEAM,
        "t": float(t) if t is not None else SUPPORT_THRESHOLD,
        "lean_resumes": resumes,
    }


def build_route_node_prior(
    weights_path: str | Path,
    *,
    backend: str = "numpy",
    allow_minilm_download: bool = False,
) -> tuple[NodePrior, dict[str, object]]:
    """Load NR weights and return a ToT ``node_prior`` plus run metadata.

    The prior returns raw feasibility scores (not a thresholded keep-set).
    """
    path = Path(weights_path)
    if not path.is_file():
        raise SystemExit(f"route-weights not found: {path}")
    meta = load_route_meta(path)
    model = load_route_model(path, backend)
    embedder = embedder_for_route_weights(
        int(model.in_dim),
        persisted_kind=str(meta.get("embedder") or ""),
        allow_download=allow_minilm_download,
    )
    probe = embedder.embed(["route-weights dim probe"])
    got_dim = int(probe.shape[1])
    expected_emb = int(model.in_dim) - EXTRA_FEATURES
    if got_dim != expected_emb:
        raise SystemExit(
            f"route-weights in_dim={model.in_dim} expects embed dim {expected_emb} "
            f"but {embedder.kind} embedder produced {got_dim}."
        )

    def node_prior(dag: ClaimDAG) -> dict[str, float]:
        emb = embed_nodes(dag, embedder)
        return feasibility_prior(model, dag, emb)

    return node_prior, meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase-1 and dump GA/GH/GI.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/cache/minif2f"),
        help="facebookresearch/miniF2F checkout with informal/",
    )
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Explicit problem ids (default: smoke 8 when --smoke)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Use the locked smoke id list: {', '.join(SMOKE_IDS)}",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--provider",
        choices=("mock", "live"),
        default="mock",
        help="mock keeps pytest-style providers; live needs API keys",
    )
    parser.add_argument(
        "--dump-root",
        type=Path,
        default=Path("data/dumps/valid"),
    )
    parser.add_argument(
        "--allow-minilm-download",
        action="store_true",
        help="Allow SentenceTransformers to download MiniLM weights",
    )
    parser.add_argument(
        "--skip-lean-compile",
        action="store_true",
        help="Set TOT_SKIP_LEAN_COMPILE=1 for the live prover",
    )
    parser.add_argument(
        "--lean-resumes",
        type=int,
        default=None,
        help="ToT resume budget after Lean fail (default 1; 0 when --skip-lean-compile)",
    )
    parser.add_argument("--k", type=int, default=None, help=f"ToT branching (default {TOT_K})")
    parser.add_argument(
        "--depth", type=int, default=None, help=f"ToT depth (default {TOT_DEPTH})"
    )
    parser.add_argument(
        "--beam", type=int, default=None, help=f"ToT beam (default {TOT_BEAM})"
    )
    parser.add_argument(
        "--t",
        type=float,
        default=None,
        help=f"Support threshold (default {SUPPORT_THRESHOLD})",
    )
    parser.add_argument(
        "--route-weights",
        type=Path,
        default=None,
        help="Optional NR npz; raw scores are added to the LLM ToT score (not thresholded).",
    )
    parser.add_argument(
        "--backend",
        choices=("numpy", "torch"),
        default="numpy",
        help="Route-weights backend (default numpy for the live prior).",
    )
    return parser


def _build_providers(kind: str, *, allow_download: bool) -> Phase1Providers:
    if kind == "mock":
        return Phase1Providers(
            reasoner=MockReasoner(include_dead=True),
            extractor=MockExtractor(),
            embedder=MockEmbedder(),
            prover=MockProver(),
        )
    if kind != "live":
        raise SystemExit(f"Unknown provider kind {kind!r} (use mock|live)")

    from tot.providers.cheap import CheapExtractor
    from tot.providers.deepseek_prover import DeepSeekProver
    from tot.providers.grok import GrokReasoner

    return Phase1Providers(
        reasoner=GrokReasoner(),
        extractor=CheapExtractor(),
        embedder=SentenceEmbedder(allow_download=allow_download),
        prover=DeepSeekProver(),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.skip_lean_compile:
        os.environ["TOT_SKIP_LEAN_COMPILE"] = "1"

    knobs = resolve_run_knobs(
        skip_lean_compile=args.skip_lean_compile,
        lean_resumes=args.lean_resumes,
        k=args.k,
        depth=args.depth,
        beam=args.beam,
        t=args.t,
    )

    ids = list(args.ids) if args.ids else (list(SMOKE_IDS) if args.smoke else None)
    problems = _select_problems(
        args.root, split=args.split, ids=ids, limit=args.limit
    )
    if not problems:
        raise SystemExit("No problems selected.")

    if args.provider == "live":
        missing = [
            name
            for name, present in [
                (
                    "GROK_API_KEY|XAI_API_KEY",
                    bool(os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")),
                ),
                (
                    "DEEPSEEK_API_KEY|OPENROUTER_API_KEY",
                    bool(
                        os.environ.get("DEEPSEEK_API_KEY")
                        or os.environ.get("DEEPSEEK_PROVER_API_KEY")
                        or os.environ.get("OPENROUTER_API_KEY")
                    ),
                ),
            ]
            if not present
        ]
        if missing:
            raise SystemExit(
                "Live provider missing env keys: "
                + ", ".join(missing)
                + ". Use --provider mock for dump wiring without keys."
            )

    providers = _build_providers(
        args.provider, allow_download=args.allow_minilm_download
    )
    node_prior = None
    route_meta: dict[str, object] = {}
    keep_tau = KEEP_THRESHOLD
    if args.route_weights is not None:
        node_prior, route_meta = build_route_node_prior(
            args.route_weights,
            backend=args.backend,
            allow_minilm_download=args.allow_minilm_download,
        )
        persisted_tau = route_meta.get("keep_threshold")
        if persisted_tau is not None:
            keep_tau = float(persisted_tau)
    args.dump_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for problem in problems:
        dump_dir = args.dump_root / problem.id
        if (dump_dir / "ga.json").is_file() and (dump_dir / "gh.json").is_file():
            print(f"{problem.id}: skip (complete dump exists)", flush=True)
            summary.append(
                {
                    "id": problem.id,
                    "skipped": True,
                    "dump_dir": str(dump_dir),
                    "k": knobs["k"],
                    "depth": knobs["depth"],
                    "beam": knobs["beam"],
                    "t": knobs["t"],
                    "lean_resumes_budget": knobs["lean_resumes"],
                }
            )
            continue
        try:
            result = run_phase1(
                problem,
                providers,
                dump_dir=dump_dir,
                k=int(knobs["k"]),
                depth=int(knobs["depth"]),
                beam=int(knobs["beam"]),
                support_t=float(knobs["t"]),
                lean_resumes=int(knobs["lean_resumes"]),
                node_prior=node_prior,
            )
        except Exception as exc:
            row = {
                "id": problem.id,
                "error": f"{type(exc).__name__}: {exc}",
                "dump_dir": str(dump_dir),
                "k": knobs["k"],
                "depth": knobs["depth"],
                "beam": knobs["beam"],
                "t": knobs["t"],
                "lean_resumes_budget": knobs["lean_resumes"],
            }
            summary.append(row)
            print(f"{problem.id}: ERROR {row['error']}")
            continue
        row = {
            "id": problem.id,
            "lean_success": lean_success(result.lean),
            "alignment_f1": node_set_f1(
                result.gi.node_id_set(), result.gh.node_id_set()
            ),
            "prune_ratio": prune_ratio(result.gi, result.ga),
            "gi_ids": sorted(result.gi.node_id_set()),
            "gh_ids": sorted(result.gh.node_id_set()),
            "dump_dir": str(dump_dir),
            "k": knobs["k"],
            "depth": knobs["depth"],
            "beam": knobs["beam"],
            "t": knobs["t"],
            "lean_resumes_budget": knobs["lean_resumes"],
        }
        if node_prior is not None:
            scores = node_prior(result.ga)
            pred = keep_set_from_scores(scores, threshold=float(keep_tau))
            row["nr_pred_ids"] = sorted(pred)
            row["keep_threshold"] = float(keep_tau)
            tree_path = dump_dir / "search_tree.json"
            if tree_path.is_file():
                sib = keep_set_from_sibling_rank(
                    load_search_tree(tree_path), scores
                ) & result.ga.node_id_set()
                if sib:
                    row["sibling_pred_ids"] = sorted(sib)
        summary.append(row)
        print(
            f"{problem.id}: lean={row['lean_success']} "
            f"F1={row['alignment_f1']:.3f} prune={row['prune_ratio']}",
            flush=True,
        )

    summary_path = args.dump_root / "summary.json"
    existing_rows: list[dict[str, object]] = []
    if summary_path.is_file():
        try:
            old = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(old, list):
                existing_rows = [row for row in old if isinstance(row, dict)]
            elif isinstance(old, dict) and isinstance(old.get("rows"), list):
                existing_rows = [row for row in old["rows"] if isinstance(row, dict)]
        except json.JSONDecodeError:
            existing_rows = []
    by_id = {str(row.get("id")): row for row in existing_rows if row.get("id")}
    for row in summary:
        by_id[str(row["id"])] = row
    merged = list(by_id.values())
    payload = {
        "knobs": knobs,
        "provider": args.provider,
        "skip_lean_compile": bool(args.skip_lean_compile),
        "route_weights": str(args.route_weights) if args.route_weights else None,
        "prior_applied": node_prior is not None,
        "n": len(merged),
        "rows": merged,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path} ({len(merged)} problems)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
