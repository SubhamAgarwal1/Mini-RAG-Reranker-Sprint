
"""Wrapper around Chroma vector store."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import chromadb
from chromadb.api.models.Collection import Collection

from . import config
from .embedding import EmbeddingModel


class _EmbeddingFunction:
    """Embedding function compatible with the Chroma interface."""

    def __init__(self, model: EmbeddingModel) -> None:
        self.model = model

    def _ensure_list(self, input: Iterable[str] | str) -> List[str]:
        if isinstance(input, str):
            return [input]
        return list(input)

    def __call__(self, input: Sequence[str]) -> List[List[float]]:  # pragma: no cover - chroma interface
        return self.model.embed(self._ensure_list(input))

    def embed_documents(self, input: Sequence[str]) -> List[List[float]]:  # pragma: no cover
        return self.model.embed(self._ensure_list(input))

    def embed_query(self, input: str | Sequence[str]) -> List[float] | List[List[float]]:  # pragma: no cover
        prepared = self._ensure_list(input)
        embeddings = self.model.embed(prepared)
        if isinstance(input, str):
            return embeddings[0]
        return embeddings

    def name(self) -> str:  # pragma: no cover - chroma interface
        return self.model.model_name


class ChromaVectorStore:
    def __init__(self, collection_name: str = "safety_chunks") -> None:
        self.model = EmbeddingModel()
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection_name = collection_name
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=_EmbeddingFunction(self.model),
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except chromadb.errors.InvalidCollectionError:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=_EmbeddingFunction(self.model),
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: Sequence[str], documents: Sequence[str], metadatas: Sequence[dict]) -> None:
        self.collection.add(ids=list(ids), documents=list(documents), metadatas=list(metadatas))

    def query(self, query: str, n_results: int) -> dict:
        return self.collection.query(query_texts=[query], n_results=n_results)
