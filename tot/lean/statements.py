"""Load Lean 4 formal statements from a pinned miniF2F Lean 4 port."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

# Recommended pin: DeepMind Lean 4 port (AlphaProof corpus). Override via env.
DEFAULT_LEAN4_GIT = "https://github.com/google-deepmind/miniF2F.git"
DEFAULT_LEAN4_ROOT = Path("data/cache/minif2f_lean4")

_THEOREM_RE = re.compile(
    r"(?P<header>(?:(?:private|protected|noncomputable)\s+)*theorem\s+(?P<name>[A-Za-z0-9_']+)\s)"
    r"(?P<body>[\s\S]*?)(?=\n(?:theorem|lemma|def|example|end|namespace|#)\b|\Z)",
)


def lean4_root() -> Path:
    override = os.environ.get("TOT_LEAN4_ROOT")
    return Path(override) if override else DEFAULT_LEAN4_ROOT


def ensure_lean4_minif2f(
    dest: str | Path | None = None,
    *,
    git_url: str = DEFAULT_LEAN4_GIT,
    rev: str | None = None,
) -> Path:
    """Clone the Lean 4 miniF2F port if missing. Returns the root path."""
    root = Path(dest) if dest is not None else lean4_root()
    if (root / "lean-toolchain").exists() or (root / "lakefile.lean").exists():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"{root} exists but is not a Lean 4 miniF2F checkout "
            "(missing lean-toolchain / lakefile.lean)."
        )
    subprocess.run(
        ["git", "clone", "--depth", "1", git_url, str(root)],
        check=True,
    )
    if rev:
        subprocess.run(["git", "fetch", "--depth", "1", "origin", rev], cwd=root, check=True)
        subprocess.run(["git", "checkout", rev], cwd=root, check=True)
    return root


def load_lean4_statements(root: str | Path | None = None) -> dict[str, str]:
    """Index ``theorem <name> ...`` snippets from ``.lean`` files under ``root``."""
    base = Path(root) if root is not None else lean4_root()
    if not base.exists():
        return {}
    out: dict[str, str] = {}
    for path in sorted(base.rglob("*.lean")):
        if path.name.startswith("."):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _THEOREM_RE.finditer(text):
            name = match.group("name")
            snippet = (match.group("header") + match.group("body")).strip()
            out.setdefault(name, snippet)
    return out


def formal_statement_for(problem_id: str, root: str | Path | None = None) -> str | None:
    index = load_lean4_statements(root)
    if problem_id in index:
        return index[problem_id]
    # Common miniF2F aliasing: mathd_* vs mathd_*
    alt = problem_id.replace("mathd_", "mathd_")
    return index.get(alt)
