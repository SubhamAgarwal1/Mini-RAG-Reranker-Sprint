"""Answer formatting utilities."""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from . import config
from .search import SearchResult

WORD_RE = re.compile(r"[A-Za-z0-9]{3,}")


def _question_terms(question: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(question)]


def _normalise_text(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii")


def _best_snippet(text: str, terms: Iterable[str], max_words: int = 30) -> str:
    cleaned = _normalise_text(text)
    lowered = cleaned.lower()
    best_index = None
    for term in terms:
        idx = lowered.find(term)
        if idx != -1 and (best_index is None or idx < best_index):
            best_index = idx
    if best_index is None:
        snippet = cleaned
    else:
        start = max(0, best_index - 80)
        end = min(len(cleaned), best_index + 220)
        snippet = cleaned[start:end]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    words = snippet.split()
    if len(words) > max_words:
        snippet = " ".join(words[:max_words]) + "..."
    return snippet


def build_answer(question: str, results: List[SearchResult]) -> Tuple[Optional[str], List[dict]]:
    contexts: List[dict] = []
    for res in results:
        contexts.append({
            "chunk_id": res.chunk_id,
            "chunk_index": res.chunk_index,
            "score": round(res.score, 4),
            "vector_score": round(res.vector_score, 4) if res.vector_score is not None else None,
            "keyword_score": round(res.keyword_score, 4) if res.keyword_score is not None else None,
            "text": res.text,
            "source_title": res.source_title,
            "source_url": res.source_url,
            "page_start": res.page_start,
            "page_end": res.page_end,
        })

    if not results:
        return None, contexts

    top_score = results[0].score
    if top_score < config.ABSTAIN_THRESHOLD:
        return None, contexts

    terms = _question_terms(question)
    snippets: List[str] = []
    citations: List[str] = []
    for res in results[:2]:
        snippet = _best_snippet(res.text, terms)
        if snippet:
            snippets.append(snippet)
            citation = f"[{res.source_title}, chunk {res.chunk_index}]"
            if citation not in citations:
                citations.append(citation)
    answer_text = " ".join(snippets)
    if citations:
        joined = " ".join(citations)
        answer_text = f"{answer_text} {joined}"
    return answer_text.strip(), contexts
