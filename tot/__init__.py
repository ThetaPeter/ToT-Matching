"""ToT dumpable claim-DAG IR, Determinator support-core, and GH matching."""

from tot.align.match import match_assignment, match_hitter
from tot.config import SUPPORT_THRESHOLD, TOT_BEAM, TOT_DEPTH, TOT_K
from tot.determinator.support import SearchBranch, compute_support, support_core
from tot.ir.dump import dump, dumps, load, loads
from tot.ir.schema import ClaimDAG, ClaimNode
from tot.ir.validate import (
    ClaimDAGValidationError,
    validate,
    validate_or_raise,
)

__all__ = [
    "SUPPORT_THRESHOLD",
    "TOT_BEAM",
    "TOT_DEPTH",
    "TOT_K",
    "ClaimDAG",
    "ClaimDAGValidationError",
    "ClaimNode",
    "SearchBranch",
    "compute_support",
    "dump",
    "dumps",
    "load",
    "loads",
    "match_assignment",
    "match_hitter",
    "support_core",
    "validate",
    "validate_or_raise",
]
