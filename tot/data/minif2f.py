"""Ingest facebookresearch/miniF2F informal proofs; drop items without a full writeup."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

MINIF2F_GIT = "https://github.com/facebookresearch/miniF2F.git"

_STUB_EXACT = re.compile(
    r"^(todo|tbd|n/?a|none|sorry|stub|pass|omitted|\.\.\.|xxx|"
    r"proof omitted|left as exercise|see solution|see above)?\.?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PairedProblem:
    id: str
    statement: str
    human_proof: str
    lean: str | None = None
    source: str = "minif2f"


def is_full_human_proof(text: str | None) -> bool:
    """Reject empty or stub informal proofs. Lean informalization is not a substitute."""
    if text is None:
        return False
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    if len(stripped.split()) < 8:
        return False
    if _STUB_EXACT.match(stripped):
        return False
    lowered = stripped.lower()
    if lowered.startswith("todo") and len(stripped) < 80:
        return False
    if lowered in {"sorry", "omit", "omitted"}:
        return False
    return True


def clone_minif2f(dest: str | Path) -> Path:
    dest = Path(dest)
    if (dest / "informal").is_dir():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise RuntimeError(
            f"{dest} is not empty and has no informal/ layout. "
            f"Unpack miniF2F there or clone into an empty directory: "
            f"git clone --depth 1 {MINIF2F_GIT} {dest}"
        )
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", MINIF2F_GIT, str(dest)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Clone failed. Unpack miniF2F locally "
            "(informal/test/*.json with informal_proof) or run: "
            f"git clone --depth 1 {MINIF2F_GIT} {dest}"
        ) from exc
    return dest


def load_paired(
    root: str | Path,
    extra_jsonl: str | Path | None = None,
) -> tuple[list[PairedProblem], list[str]]:
    """Return (kept paired problems, discarded ids). Does not use Lean as GA′."""
    root = Path(root)
    kept: list[PairedProblem] = []
    discarded: list[str] = []
    seen: set[str] = set()
    lean_index = _load_lean_index(root)

    for problem in _iter_root_problems(root, lean_index):
        _accumulate(problem, kept, discarded, seen)

    if extra_jsonl is not None:
        for problem in _iter_jsonl(Path(extra_jsonl), source="olympiad_jsonl"):
            _accumulate(problem, kept, discarded, seen)

    return kept, discarded


def _accumulate(
    problem: PairedProblem,
    kept: list[PairedProblem],
    discarded: list[str],
    seen: set[str],
) -> None:
    if problem.id in seen:
        return
    seen.add(problem.id)
    if is_full_human_proof(problem.human_proof):
        kept.append(problem)
    else:
        discarded.append(problem.id)


def _informal_root(root: Path) -> Path:
    informal = root / "informal"
    if informal.is_dir():
        return informal
    return root


def _iter_root_problems(root: Path, lean_index: dict[str, str]) -> Iterable[PairedProblem]:
    base = _informal_root(root)
    if not base.exists():
        return
    paths = sorted(
        p
        for p in base.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".md", ".markdown"}
    )
    for path in paths:
        problem = _parse_file(path, lean_index)
        if problem is not None:
            yield problem


def _parse_file(path: Path, lean_index: dict[str, str]) -> PairedProblem | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        pid = str(
            obj.get("problem_name")
            or obj.get("id")
            or obj.get("name")
            or path.stem
        )
        statement = str(
            obj.get("informal_statement") or obj.get("statement") or ""
        )
        proof = obj.get("informal_proof")
        if proof is None:
            proof = obj.get("human_proof")
        if proof is None:
            proof = obj.get("proof")
        proof_s = "" if proof is None else str(proof)
        lean = obj.get("lean")
        if lean is None:
            lean = obj.get("formal_statement")
        lean_s = None if lean is None else str(lean)
        if not lean_s:
            lean_s = lean_index.get(pid)
        return PairedProblem(
            id=pid,
            statement=statement,
            human_proof=proof_s,
            lean=lean_s,
            source="minif2f",
        )
    pid, statement, proof = _parse_markdown(text, path.stem)
    return PairedProblem(
        id=pid,
        statement=statement,
        human_proof=proof,
        lean=lean_index.get(pid),
        source="minif2f",
    )


def _parse_markdown(text: str, default_id: str) -> tuple[str, str, str]:
    pid = default_id
    name_m = re.search(
        r"^(?:problem_name|id|name)\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE
    )
    if name_m:
        pid = name_m.group(1).strip().strip("`\"'")
    statement = _field_or_section(
        text, ("informal_statement", "statement"), ("informal statement", "statement")
    )
    proof = _field_or_section(
        text,
        ("informal_proof", "human_proof", "proof"),
        ("informal proof", "human proof", "proof"),
    )
    return pid, statement, proof


def _field_or_section(
    text: str,
    field_names: tuple[str, ...],
    headings: tuple[str, ...],
) -> str:
    for name in field_names:
        m = re.search(
            rf"^{name}\s*:\s*(.+)$",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if m:
            return m.group(1).strip()
    lower = text.splitlines()
    for i, line in enumerate(lower):
        stripped = line.strip().lstrip("#").strip().lower()
        if stripped in headings:
            chunk: list[str] = []
            for nxt in lower[i + 1 :]:
                if nxt.strip().startswith("#"):
                    break
                chunk.append(nxt)
            return "\n".join(chunk).strip()
    return ""


def _iter_jsonl(path: Path, source: str) -> Iterable[PairedProblem]:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        pid = str(obj.get("id") or obj.get("problem_name") or obj.get("name") or "")
        if not pid:
            continue
        statement = str(obj.get("statement") or obj.get("informal_statement") or "")
        proof = str(obj.get("human_proof") or obj.get("informal_proof") or "")
        lean = obj.get("lean")
        yield PairedProblem(
            id=pid,
            statement=statement,
            human_proof=proof,
            lean=None if lean is None else str(lean),
            source=source,
        )


def _load_lean_index(root: Path) -> dict[str, str]:
    """Optional theorem snippets keyed by name. Never used as a human proof."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*.lean")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in re.finditer(
            r"theorem\s+([A-Za-z0-9_']+)\s*[\s\S]{0,400}",
            text,
        ):
            name = match.group(1)
            snippet = match.group(0).strip()
            out.setdefault(name, snippet)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count miniF2F items that have a full human informal proof."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--extra-jsonl", type=Path, default=None)
    parser.add_argument(
        "--clone",
        action="store_true",
        help=f"git clone --depth 1 {MINIF2F_GIT} into --root (optional, heavy)",
    )
    args = parser.parse_args(argv)
    root = args.root
    if args.clone:
        clone_minif2f(root)
    kept, discarded = load_paired(root, extra_jsonl=args.extra_jsonl)
    print(f"paired N={len(kept)}")
    print(f"discarded count={len(discarded)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
