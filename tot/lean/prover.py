"""Lean from GI only. Failures become a new informal proof state; GI is never mutated."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tot.ir.dump import dumps
from tot.ir.schema import ClaimDAG
from tot.prove.nl_proof import serialize_gi
from tot.providers.base import Prover


@dataclass
class LeanResult:
    ok: bool
    code: str
    errors: list[str]
    new_proof_state: str


def run_prover(prover: Prover, gi: ClaimDAG) -> LeanResult:
    """Call the prover with a GI-only payload. Do not write back into ``gi``."""
    snapshot = dumps(gi)
    payload = serialize_gi(gi)
    raw = prover.prove(payload)
    if dumps(gi) != snapshot:
        raise RuntimeError("Lean adapter mutated GI; GI must be left unchanged")
    ok = bool(raw.get("ok"))
    code = str(raw.get("code") or "")
    errors = _as_error_list(raw.get("errors"))
    new_state = "" if ok else errors_to_proof_state(errors, gi)
    if dumps(gi) != snapshot:
        raise RuntimeError("Lean adapter mutated GI; GI must be left unchanged")
    return LeanResult(ok=ok, code=code, errors=errors, new_proof_state=new_state)


def errors_to_proof_state(errors: Sequence[str], gi: ClaimDAG) -> str:
    """Parse Lean errors/goals into an informal proof state for ToT resume."""
    err_text = "\n".join(str(item) for item in errors) or "(no error text)"
    claims = "; ".join(f"{node.id}: {node.statement}" for node in gi.nodes)
    return (
        "Lean failed. Remaining goals/errors:\n"
        f"{err_text}\n"
        "Continue the informal proof from this state. "
        "Do not rewrite GI.\n"
        f"Current GI claims: {claims}"
    )


def _as_error_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]
