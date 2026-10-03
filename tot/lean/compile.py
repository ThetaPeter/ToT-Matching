"""Lean 4 compile helper. Real ``lake``/``lean`` when available; injectable for tests."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

CompileFn = Callable[[str, str | None], "CompileResult"]


@dataclass
class CompileResult:
    ok: bool
    errors: list[str]
    stdout: str = ""
    stderr: str = ""


def lean_available() -> bool:
    return shutil.which("lake") is not None or shutil.which("lean") is not None


def compile_lean_code(
    code: str,
    *,
    formal_statement: str | None = None,
    work_dir: str | Path | None = None,
    timeout_s: float = 120.0,
) -> CompileResult:
    """Typecheck Lean 4 code. Prefer ``lake env lean`` inside a Lake project."""
    lean_bin = shutil.which("lean")
    lake_bin = shutil.which("lake")
    if lean_bin is None and lake_bin is None:
        return CompileResult(
            ok=False,
            errors=[
                "Lean 4 toolchain not found on PATH (need `lean` or `lake`). "
                "Install elan + Lean 4, or pass a custom compile_fn."
            ],
        )

    text = code.strip()
    if formal_statement and "theorem" not in text and "example" not in text:
        text = formal_statement.rstrip() + "\n" + text + "\n"

    if work_dir is not None:
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        target = root / "_tot_check.lean"
        target.write_text(text, encoding="utf-8")
        return _run_lean(target, lake_bin=lake_bin, lean_bin=lean_bin, timeout_s=timeout_s)

    with tempfile.TemporaryDirectory(prefix="tot_lean_") as tmp:
        target = Path(tmp) / "check.lean"
        target.write_text(text, encoding="utf-8")
        return _run_lean(target, lake_bin=lake_bin, lean_bin=lean_bin, timeout_s=timeout_s)


def _run_lean(
    path: Path,
    *,
    lake_bin: str | None,
    lean_bin: str | None,
    timeout_s: float,
) -> CompileResult:
    cwd = path.parent
    if lake_bin and (cwd / "lakefile.lean").exists() or (cwd / "lakefile.toml").exists():
        cmd = [lake_bin, "env", "lean", str(path)]
    elif lake_bin and Path(os.environ.get("TOT_LEAN4_ROOT", "")).exists():
        cmd = [lake_bin, "env", "lean", str(path)]
        cwd = Path(os.environ["TOT_LEAN4_ROOT"])
    elif lean_bin:
        cmd = [lean_bin, str(path)]
    elif lake_bin:
        cmd = [lake_bin, "env", "lean", str(path)]
    else:
        return CompileResult(ok=False, errors=["No lean/lake binary available"])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except OSError as exc:
        return CompileResult(ok=False, errors=[f"Failed to run {cmd[0]}: {exc}"])
    except subprocess.TimeoutExpired:
        return CompileResult(ok=False, errors=[f"Lean compile timed out after {timeout_s}s"])

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode == 0:
        return CompileResult(ok=True, errors=[], stdout=stdout, stderr=stderr)
    errors = _parse_errors(stdout, stderr)
    if not errors:
        errors = [stderr.strip() or stdout.strip() or f"lean exit {proc.returncode}"]
    return CompileResult(ok=False, errors=errors, stdout=stdout, stderr=stderr)


def _parse_errors(stdout: str, stderr: str) -> list[str]:
    blob = "\n".join(part for part in (stdout, stderr) if part)
    lines = []
    for line in blob.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"error|unsolved goals|failed", stripped, re.IGNORECASE):
            lines.append(stripped)
    return lines[:50]
