
"""Run baseline vs hybrid comparison over the question set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tabulate import tabulate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qaservice import config
from src.qaservice.service import QAService


def load_questions(path: Path) -> list[str]:
    questions: list[str] = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            questions.append(payload["question"])
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        questions = [entry["question"] for entry in data]
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and hybrid retrieval modes")
    parser.add_argument("--output", type=Path, help="Optional path to save the table")
    args = parser.parse_args()

    qa_service = QAService()
    questions_path = config.QUESTIONS_JSONL
    if not questions_path.exists():
        raise SystemExit(f"Question file not found: {questions_path}")

    questions = load_questions(questions_path)
    rows = []
    for idx, question in enumerate(questions, start=1):
        baseline = qa_service.ask(question, k=config.ANSWER_TOP_K, mode="baseline")
        hybrid = qa_service.ask(question, k=config.ANSWER_TOP_K, mode="hybrid")
        base_score = baseline["contexts"][0]["score"] if baseline["contexts"] else 0.0
        hybrid_score = hybrid["contexts"][0]["score"] if hybrid["contexts"] else 0.0
        rows.append(
            [
                idx,
                question,
                round(base_score, 3),
                round(hybrid_score, 3),
                baseline["answer"] or "<abstained>",
                hybrid["answer"] or "<abstained>",
            ]
        )

    headers = ["#", "question", "baseline_score", "hybrid_score", "baseline_answer", "hybrid_answer"]
    table = tabulate(rows, headers=headers, tablefmt="github")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
    else:
        print(table)


if __name__ == "__main__":
    main()
