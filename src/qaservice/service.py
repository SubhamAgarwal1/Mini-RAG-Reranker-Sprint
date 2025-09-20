
"""High-level question answering service."""
from __future__ import annotations

from typing import Dict

from . import config
from .answers import build_answer
from .search import SafetySearchEngine


class QAService:
    def __init__(self) -> None:
        self.engine = SafetySearchEngine()

    def ask(self, question: str, k: int, mode: str) -> Dict[str, object]:
        k = max(1, min(k, 20))
        mode = mode or "hybrid"
        mode = mode.lower()
        results = self.engine.search(question, k, mode)
        answer, contexts = build_answer(question, results)
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reranker_used": mode == "hybrid",
            "mode": mode,
        }
