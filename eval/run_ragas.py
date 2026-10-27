"""Run the agent over eval/questions.jsonl and score it with Ragas.  [Phase 1 baseline onward]

Steps:
1. Load questions with a reference answer (others are skipped and reported).
2. Run build_graph().invoke() per question; keep question, contexts, answer, retrievers.
3. Score with Ragas (context precision/recall, faithfulness, answer relevancy,
   factual correctness) using EVAL_JUDGE_MODEL as the judge.
4. Add custom metrics: router accuracy, Text2Cypher result-set match.
5. Write eval/results/<timestamp>.json and print scores per question type.
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent


def load_questions(path: Path = EVAL_DIR / "questions.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    questions = load_questions()
    ready = [q for q in questions if q.get("reference_answer")]
    print(f"{len(ready)}/{len(questions)} questions have reference answers.")
    if not ready:
        print("Add reference answers to eval/questions.jsonl before running Ragas.")
        return
    raise NotImplementedError("Phase 1: run agent and score with Ragas")


if __name__ == "__main__":
    main()
