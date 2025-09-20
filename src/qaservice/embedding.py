"""Embedding helpers built on top of sentence-transformers."""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from . import config


class EmbeddingModel:
    """Wrap a sentence-transformers model with predictable behaviour."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBED_MODEL
        self._model: SentenceTransformer | None = None
        self._ensure_seed()

    @staticmethod
    def _ensure_seed() -> None:
        np.random.seed(config.SEED)
        torch.manual_seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """Compute embeddings for a list of texts."""
        text_list = list(texts)
        if not text_list:
            return []
        embeddings = self.model.encode(
            text_list,
            batch_size=config.EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
