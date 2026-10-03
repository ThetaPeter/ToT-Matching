from tot.prove.nl_proof import build_gi_payload, prove_nl, serialize_gi

__all__ = [
    "NCPConfig",
    "NCPReasoner",
    "build_gi_payload",
    "load_ncp",
    "prove_nl",
    "save_ncp",
    "serialize_gi",
    "train_ncp",
]

_NCP_EXPORTS = {"NCPConfig", "NCPReasoner", "load_ncp", "save_ncp", "train_ncp"}


def __getattr__(name: str):
    if name in _NCP_EXPORTS:
        from tot.prove import ncp

        return getattr(ncp, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
