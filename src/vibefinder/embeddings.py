"""Embedding provider interfaces for retrieval tools."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbeddingProvider(Protocol):
    """Provider-agnostic text embedding interface."""

    @property
    def name(self) -> str:
        """Stable provider/model identifier."""

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed text inputs into a 2D float32 matrix."""


class SentenceTransformerEmbeddingProvider:
    """SentenceTransformers-backed local embedding provider."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 32,
        device: str | None = None,
        show_progress_bar: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.show_progress_bar = show_progress_bar
        self._model = None

    @property
    def name(self) -> str:
        return self.model_name

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        model = self.load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress_bar,
        )
        return normalize_embedding_matrix(embeddings)

    def load_model(self):
        """Load and cache the underlying SentenceTransformers model."""

        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for the default lyric embedding provider. "
                "Install project dependencies or pass a TextEmbeddingProvider to ToolContext."
            ) from exc

        if self.device is None:
            self._model = SentenceTransformer(self.model_name)
        else:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


@lru_cache(maxsize=1)
def get_default_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    """Return the configured default embedding provider."""

    model_name = os.getenv("VIBEFINDER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    batch_size = int(os.getenv("VIBEFINDER_EMBEDDING_BATCH_SIZE", "32"))
    device = os.getenv("VIBEFINDER_EMBEDDING_DEVICE") or None
    return SentenceTransformerEmbeddingProvider(
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )


def normalize_embedding_matrix(values: np.ndarray) -> np.ndarray:
    """Validate and L2-normalize an embedding matrix."""

    matrix = np.asarray(values, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError("Embedding provider must return a 2D matrix.")
    if matrix.shape[1] == 0:
        raise ValueError("Embedding provider returned zero-dimensional vectors.")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")
