"""MiniLM-class embedder with a hashed fallback (no download required)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tot.providers.base import hashed_token_vectors

DEFAULT_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIM = 384
FALLBACK_DIM = 32


def _resolved_model_name(model_name: str) -> str:
    """Prefer ``TOT_MINILM_PATH`` when it points at a local MiniLM checkout."""
    override = os.environ.get("TOT_MINILM_PATH", "").strip()
    if override:
        path = Path(override)
        if path.is_dir():
            return str(path)
    return model_name


def hashed_embeddings(texts: Sequence[str], dim: int = FALLBACK_DIM) -> np.ndarray:
    return hashed_token_vectors(texts, dim=dim)


class SentenceEmbedder:
    """Use MiniLM when the package and local weights exist; otherwise hash vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MINILM,
        *,
        allow_download: bool = False,
        force_hashed: bool = False,
    ) -> None:
        self.model_name = _resolved_model_name(model_name)
        self.allow_download = allow_download
        self.force_hashed = bool(force_hashed)
        self._model = None
        self._failed = bool(force_hashed)

    @property
    def kind(self) -> str:
        """``minilm`` or ``hashed``. Valid after ``embed`` / ``_try_model``."""
        if self.force_hashed:
            return "hashed"
        if self._try_model() is not None:
            return "minilm"
        return "hashed"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            dim = MINILM_DIM if self.kind == "minilm" else FALLBACK_DIM
            return np.zeros((0, dim), dtype=float)
        model = self._try_model()
        if model is None:
            return hashed_embeddings(texts)
        encoded = model.encode(list(texts), convert_to_numpy=True)
        return np.asarray(encoded, dtype=float)

    def _try_model(self):
        if self._failed:
            return None
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._failed = True
            return None
        try:
            kwargs: dict[str, object] = {}
            local_dir = Path(self.model_name).is_dir()
            if local_dir or not self.allow_download:
                kwargs["local_files_only"] = True
            self._model = SentenceTransformer(self.model_name, **kwargs)
        except Exception:
            self._failed = True
            return None
        return self._model
