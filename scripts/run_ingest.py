
"""Run the ingestion pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qaservice.ingest import PdfIngestor


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into SQLite and Chroma")
    parser.parse_args()
    PdfIngestor().run()


if __name__ == "__main__":
    main()
