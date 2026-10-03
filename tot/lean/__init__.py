"""Lean package exports."""

from tot.lean.compile import CompileFn, CompileResult, compile_lean_code, lean_available
from tot.lean.prover import LeanResult, errors_to_proof_state, run_prover
from tot.lean.statements import (
    ensure_lean4_minif2f,
    formal_statement_for,
    load_lean4_statements,
)

__all__ = [
    "CompileFn",
    "CompileResult",
    "LeanResult",
    "compile_lean_code",
    "ensure_lean4_minif2f",
    "errors_to_proof_state",
    "formal_statement_for",
    "lean_available",
    "load_lean4_statements",
    "run_prover",
]
