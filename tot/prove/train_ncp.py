"""Train N_{C,P} on dumped GI JSON + human writeups (valid split)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tot.align.embed import SentenceEmbedder
from tot.data.minif2f import load_paired
from tot.ir.dump import load
from tot.prove.ncp import NCPConfig, NCPExample, save_ncp, train_ncp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-root", type=Path, default=Path("data/dumps/valid"))
    parser.add_argument("--minif2f-root", type=Path, default=Path("data/cache/minif2f"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub decoder (no HuggingFace T5 download)",
    )
    parser.add_argument("--t5-name", default="t5-base")
    parser.add_argument("--out", type=Path, default=Path("data/dumps/valid/ncp.pt"))
    parser.add_argument("--allow-minilm-download", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    kept, _ = load_paired(args.minif2f_root)
    by_id = {problem.id: problem for problem in kept}
    examples: list[NCPExample] = []
    embedder = SentenceEmbedder(allow_download=args.allow_minilm_download)

    for gi_path in sorted(args.dump_root.glob("*/gi.json")):
        problem_id = gi_path.parent.name
        if problem_id not in by_id:
            continue
        gi = load(gi_path)
        texts = [
            f"{node.id}: {node.statement} {node.natural_language}" for node in gi.nodes
        ]
        mat = embedder.embed(texts)
        emb = {
            node.id: np.asarray(mat[i], dtype=np.float32)
            for i, node in enumerate(gi.nodes)
        }
        examples.append(
            NCPExample(
                gi=gi,
                human_proof=by_id[problem_id].human_proof,
                embeddings=emb,
            )
        )
        if args.limit is not None and len(examples) >= args.limit:
            break

    if not examples:
        raise SystemExit(f"No GI dumps with matching writeups under {args.dump_root}")

    config = NCPConfig(stub_decoder=bool(args.stub), t5_name=args.t5_name)
    model = train_ncp(examples, epochs=args.epochs, config=config)
    save_ncp(model, args.out)
    meta = {
        "n_examples": len(examples),
        "stub_decoder": bool(args.stub),
        "t5_name": args.t5_name,
        "epochs": args.epochs,
        "ids": [ex.gi.problem_id for ex in examples],
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Trained NCP on {len(examples)} GI dumps → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
