
"""Retrieval and reranking logic."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from . import config, db
from .vectorstore import ChromaVectorStore

TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")

STOP_WORDS = {
    'the','and','for','with','that','from','this','into','about','what','when','where','how','does','should','are','was','were','have','has','will','would','can','could','may','might','which','using','according','please','provide','explain','describe','does','do','at','of','in','to','on','by','or','an','any','be','a','as','is','it','their','there','who','whom'
}

@dataclass
class SearchResult:
    chunk_id: int
    source_id: int
    chunk_index: int
    text: str
    score: float
    vector_score: Optional[float]
    keyword_score: Optional[float]
    page_start: Optional[int]
    page_end: Optional[int]
    source_title: str
    source_url: str


class SafetySearchEngine:
    def __init__(self) -> None:
        self.vector_store = ChromaVectorStore()

    @staticmethod
    def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        values = list(scores.values())
        if len(set(values)) == 1:
            return {key: 1.0 for key in scores}
        min_score = min(values)
        max_score = max(values)
        span = max_score - min_score
        return {key: (value - min_score) / span for key, value in scores.items()}

    def _build_results(self, rows, source_map, score_map, vector_map=None, keyword_map=None) -> List[SearchResult]:
        results: List[SearchResult] = []
        for row in rows:
            src = source_map[row["source_id"]]
            results.append(
                SearchResult(
                    chunk_id=row["id"],
                    source_id=row["source_id"],
                    chunk_index=row["chunk_index"],
                    text=row["text"],
                    score=score_map.get(row["id"], 0.0),
                    vector_score=(vector_map or {}).get(row["id"]),
                    keyword_score=(keyword_map or {}).get(row["id"]),
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    source_title=src["title"],
                    source_url=src["url"],
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        response = self.vector_store.query(query, n_results=top_k)
        ids = response.get("ids", [[]])[0]
        distances = response.get("distances", [[]])[0]
        if not ids:
            return []
        chunk_ids = [int(identifier.split("-")[1]) for identifier in ids]
        similarity_scores = {
            chunk_id: 1 - dist if dist is not None else 0.0
            for chunk_id, dist in zip(chunk_ids, distances)
        }

        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, source_id, chunk_index, text, page_start, page_end FROM chunks WHERE id IN (%s)" %
                ",".join("?" for _ in chunk_ids),
                chunk_ids,
            ).fetchall()
            source_map = db.fetch_source_map(conn)

        return self._build_results(rows, source_map, similarity_scores, vector_map=similarity_scores)

    def lexical_search(self, query: str, top_k: int) -> List[SearchResult]:
        tokens = [tok for tok in TOKEN_RE.findall(query.lower()) if tok not in STOP_WORDS]
        if not tokens:
            return []
        fts_query = " OR ".join(tokens)
        bm25_scores: Dict[int, float] = {}
        rows = []
        with db.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT chunks.id, chunks.source_id, chunks.chunk_index, chunks.text, chunks.page_start, chunks.page_end,
                       bm25(chunk_fts) AS bm25_score
                FROM chunk_fts
                JOIN chunks ON chunk_fts.rowid = chunks.id
                WHERE chunk_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (fts_query, top_k),
            )
            rows = cursor.fetchall()
            source_map = db.fetch_source_map(conn)

        for row in rows:
            raw_score = row["bm25_score"]
            if raw_score is None or math.isinf(raw_score):
                continue
            bm25_scores[row["id"]] = -raw_score

        return self._build_results(rows, source_map, bm25_scores, keyword_map=bm25_scores)

    def hybrid_search(self, query: str, top_k: int, alpha: float = config.HYBRID_ALPHA) -> List[SearchResult]:
        vector_candidates = self.vector_search(query, config.VECTOR_TOP_K)
        lexical_candidates = self.lexical_search(query, config.LEXICAL_TOP_K)

        vector_scores = {res.chunk_id: res.vector_score or 0.0 for res in vector_candidates}
        keyword_scores = {res.chunk_id: res.keyword_score or 0.0 for res in lexical_candidates}

        norm_vector = self._normalize(vector_scores)
        norm_keyword = self._normalize(keyword_scores)

        all_ids = set(norm_vector) | set(norm_keyword)
        if not all_ids:
            return []

        with db.get_connection() as conn:
            rows = conn.execute(
                "SELECT id, source_id, chunk_index, text, page_start, page_end FROM chunks WHERE id IN (%s)" %
                ",".join("?" for _ in all_ids),
                list(all_ids),
            ).fetchall()
            source_map = db.fetch_source_map(conn)

        combined_scores = {
            chunk_id: alpha * norm_vector.get(chunk_id, 0.0) + (1 - alpha) * norm_keyword.get(chunk_id, 0.0)
            for chunk_id in all_ids
        }
        return self._build_results(
            rows,
            source_map,
            combined_scores,
            vector_map=vector_scores,
            keyword_map=keyword_scores,
        )[:top_k]

    def search(self, query: str, k: int, mode: str) -> List[SearchResult]:
        if mode == "baseline":
            return self.vector_search(query, k)
        if mode == "hybrid":
            return self.hybrid_search(query, k)
        raise ValueError("mode must be 'baseline' or 'hybrid'")
