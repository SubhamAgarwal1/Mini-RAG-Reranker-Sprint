"""Configuration for the Q&A service."""
from pathlib import Path

# Root directory for the repository
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
SQLITE_DB = DATA_DIR / "sqlite" / "chunks.db"
CHROMA_DIR = INDEX_DIR / "chroma"

SOURCES_JSON = ROOT_DIR / "sources.json"
QUESTIONS_JSONL = ROOT_DIR / "data" / "questions.jsonl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 16
SEED = 42

# Chunking parameters
CHUNK_CHAR_TARGET = 900
CHUNK_CHAR_OVERLAP = 100
MIN_CHUNK_CHAR = 200

# Retrieval defaults
VECTOR_TOP_K = 30
LEXICAL_TOP_K = 30
HYBRID_ALPHA = 0.7
ANSWER_TOP_K = 4
ABSTAIN_THRESHOLD = 0.28
