
"""FastAPI application exposing the /ask endpoint."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.qaservice import config
from src.qaservice.service import QAService


class AskRequest(BaseModel):
    q: str = Field(..., description="Natural language question")
    k: int = Field(default=config.ANSWER_TOP_K, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(baseline|hybrid)$")


class Context(BaseModel):
    chunk_id: int
    chunk_index: int
    score: float
    vector_score: float | None
    keyword_score: float | None
    text: str
    source_title: str
    source_url: str
    page_start: int | None
    page_end: int | None


class AskResponse(BaseModel):
    question: str
    answer: str | None
    contexts: list[Context]
    reranker_used: bool
    mode: str


app = FastAPI(title="Industrial Safety QA Service", version="1.0.0")
_service: QAService | None = None


def get_service() -> QAService:
    global _service
    if _service is None:
        _service = QAService()
    return _service


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.q.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    payload = get_service().ask(question, request.k, request.mode)
    return AskResponse(**payload)


@app.get("/health", response_model=dict)
def health() -> dict:
    return {"status": "ok"}
