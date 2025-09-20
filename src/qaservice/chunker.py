"""Utilities for chunking PDF text."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from . import config

WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Collapse whitespace and trim text."""
    return WHITESPACE_RE.sub(" ", text).strip()


def split_paragraphs(text: str) -> List[str]:
    """Split a block of text into normalized paragraphs."""
    blocks: List[str] = []
    for raw in re.split(r"\n\s*\n", text):
        cleaned = clean_text(raw)
        if cleaned:
            blocks.append(cleaned)
    return blocks


def chunk_paragraphs(
    paragraphs: Sequence[str],
    target_chars: int = config.CHUNK_CHAR_TARGET,
    min_chars: int = config.MIN_CHUNK_CHAR,
    overlap_paras: int = 1,
) -> Iterable[str]:
    """Yield chunks built from consecutive paragraphs."""
    if not paragraphs:
        return

    buffer: List[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        if current_len and current_len + 1 + para_len > target_chars and current_len >= min_chars:
            yield " ".join(buffer)
            overlap = buffer[-overlap_paras:]
            buffer = overlap.copy()
            current_len = len(" ".join(buffer)) if buffer else 0
        buffer.append(para)
        current_len = len(" ".join(buffer))

        while current_len > target_chars * 1.5:
            text = " ".join(buffer)
            midpoint = target_chars
            chunk = text[:midpoint]
            yield chunk.strip()
            remainder = text[midpoint:]
            buffer = [remainder.strip()] if remainder.strip() else []
            current_len = len(" ".join(buffer)) if buffer else 0

    if buffer:
        yield " ".join(buffer)
