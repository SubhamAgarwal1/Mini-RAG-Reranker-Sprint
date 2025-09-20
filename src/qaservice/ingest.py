
"""Ingest PDFs: chunk text, store in SQLite, and index in Chroma."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pypdf
from tqdm import tqdm

from . import config, chunker, db
from .vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class ChunkPayload:
    source_id: int
    chunk_index: int
    text: str
    page_start: int | None
    page_end: int | None

    @property
    def record(self) -> Dict[str, object]:
        return {
            "source_id": self.source_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_len": len(self.text),
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


class PdfIngestor:
    def __init__(self) -> None:
        self.vector_store = ChromaVectorStore()

    def run(self) -> None:
        sources = db.load_sources()
        path_map = self._validate_sources(sources)

        with db.get_connection() as conn:
            db.initialise_database(conn)
            db.upsert_sources(conn, sources)
            file_to_source_id = self._map_source_ids(conn)

            if not file_to_source_id:
                logger.error("No sources found to ingest.")
                return

            all_chunks: List[ChunkPayload] = []
            for file_name, resolved_path in tqdm(path_map.items(), desc="Chunking PDFs"):
                source_id = file_to_source_id.get(file_name)
                if source_id is None:
                    logger.warning("Skipping %s because it is missing in the sources table", file_name)
                    continue
                chunks = list(self._chunk_pdf(resolved_path, source_id))
                all_chunks.extend(chunks)

            logger.info("Chunked %d pieces from %d PDFs", len(all_chunks), len(path_map))

            db.insert_chunks(conn, (payload.record for payload in all_chunks))

            rows = conn.execute(
                "SELECT id, source_id, chunk_index, text, page_start, page_end FROM chunks ORDER BY id"
            ).fetchall()

        self.vector_store.reset()
        ids = [f"chunk-{row['id']}" for row in rows]
        documents = [row["text"] for row in rows]
        metadata = [
            {
                "chunk_id": int(row["id"]),
                "source_id": int(row["source_id"]),
                "chunk_index": int(row["chunk_index"]),
                "page_start": row["page_start"],
                "page_end": row["page_end"],
            }
            for row in rows
        ]

        logger.info("Indexing %d chunks in Chroma", len(ids))
        self.vector_store.add(ids=ids, documents=documents, metadatas=metadata)
        logger.info("Ingestion complete")

    @staticmethod
    def _validate_sources(sources: List[Dict[str, str]]) -> Dict[str, Path]:
        missing: List[str] = []
        mapping: Dict[str, Path] = {}
        for entry in sources:
            file_name = entry.get("file_name")
            if not file_name:
                raise ValueError("Each source entry must include 'file_name'")
            pdf_path = config.RAW_DIR / file_name
            if pdf_path.exists():
                mapping[file_name] = pdf_path
            else:
                missing.append(file_name)
        if missing:
            logger.warning("Missing %d PDFs referenced in sources.json: %s", len(missing), ", ".join(missing))
        return mapping

    @staticmethod
    def _map_source_ids(conn) -> Dict[str, int]:
        cursor = conn.execute("SELECT id, file_name FROM sources")
        return {row["file_name"]: row["id"] for row in cursor.fetchall()}

    @staticmethod
    def _chunk_pdf(pdf_path: Path, source_id: int) -> Iterable[ChunkPayload]:
        reader = pypdf.PdfReader(str(pdf_path))
        chunk_index = 0
        buffer: List[str] = []
        buffer_start_page: int | None = None
        last_page_seen: int | None = None

        for page_number, page in enumerate(reader.pages, start=1):
            last_page_seen = page_number
            text = page.extract_text() or ""
            paragraphs = chunker.split_paragraphs(text)
            if not paragraphs:
                continue
            for para in paragraphs:
                if not buffer:
                    buffer_start_page = page_number
                buffer.append(para)
                combined = " ".join(buffer)
                if len(combined) >= config.CHUNK_CHAR_TARGET:
                    yield ChunkPayload(
                        source_id=source_id,
                        chunk_index=chunk_index,
                        text=combined,
                        page_start=buffer_start_page,
                        page_end=page_number,
                    )
                    chunk_index += 1
                    buffer = buffer[-1:]
                    buffer_start_page = page_number
            combined = " ".join(buffer)
            if len(combined) > config.CHUNK_CHAR_TARGET * 1.5:
                yield ChunkPayload(
                    source_id=source_id,
                    chunk_index=chunk_index,
                    text=combined,
                    page_start=buffer_start_page,
                    page_end=page_number,
                )
                chunk_index += 1
                buffer = []
                buffer_start_page = None

        if buffer:
            yield ChunkPayload(
                source_id=source_id,
                chunk_index=chunk_index,
                text=" ".join(buffer),
                page_start=buffer_start_page,
                page_end=last_page_seen,
            )
