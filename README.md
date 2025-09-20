# Industrial Safety QA Service

This project builds a small question-answering service over a bundle of industrial and machine safety PDFs. It ingests the provided documents, chunks and embeds them with a local SentenceTransformer, stores the text in SQLite with an FTS5 sidecar, and serves answers through a FastAPI `/ask` endpoint. A lightweight hybrid reranker blends dense cosine scores with keyword matches so evidence with exact terminology surfaces ahead of generic paragraphs.

## Setup

Requirements: Python 3.10+, CPU only.

```bash
python -m venv .venv
. .venv/Scripts/activate  # use source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

The data pack is already unpacked under `data/raw/`. Sources and URLs are tracked in `sources.json`; the eight evaluation questions live in `data/questions.jsonl`.

## Build the indexes

Run the ingestion script once after installing dependencies. It extracts text from the PDFs, writes chunk metadata to `data/sqlite/chunks.db`, and persists a Chroma collection in `data/index/chroma/`.

```bash
python scripts/run_ingest.py
```

The first run takes ~4 minutes on CPU because ~2.3k chunks are embedded.

## Serve the API

Launch the FastAPI app with Uvicorn. The service keeps the SQLite connection and Chroma collection in memory.

```bash
uvicorn app:app --reload --port 8000
```

Then ask questions with the single POST endpoint:

### Example: easy
```bash
curl -s -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{"q":"According to OSHA 3170, what guarding methods are recommended for preventing amputations at the point of operation?","k":3,"mode":"hybrid"}'
```

### Example: trickier compliance question
```bash
curl -s -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{"q":"How should designers apply the risk graph in EN ISO 13849-1 to determine required performance level?","k":3,"mode":"hybrid"}'
```

The response includes a short extractive answer (trimmed to ~30 words), a citation block for each chunk, and a `reranker_used` flag indicating whether hybrid blending was applied.

## Baseline vs hybrid comparison

Run the evaluation helper to reproduce the before/after table for the eight provided questions:

```bash
python scripts/run_eval.py --output data/eval_table.md
```

The Markdown table at `data/eval_table.md` captures both scores and answers. With the current configuration the hybrid reranker improved 7/8 questions and lifted the average top-1 score from 0.64 to 0.80.

## Project layout

- `data/raw/` – supplied PDFs (20 files)
- `data/sqlite/chunks.db` – chunk text + metadata with FTS5 search
- `data/index/chroma/` – Chroma persistent collection for cosine search
- `src/qaservice/` – core modules (chunking, ingestion, retrieval, answering, FastAPI glue)
- `scripts/run_ingest.py` – one-shot ingestion
- `scripts/run_eval.py` – baseline vs hybrid comparison
- `data/questions.jsonl` – eight evaluation prompts

## What I learned

1. Small domain packs still need careful normalization: without collapsing ligatures and bullet artifacts the answers ballooned with useless boilerplate. A narrow, rule-based summariser keeps responses grounded and short without an LLM.
2. SQLite FTS5 plus Chroma was a pragmatic mix – handling both dense and lexical signals locally. The hybrid reranker only required a few lines once the scores were normalised and shared chunk IDs were tracked in both stores.
