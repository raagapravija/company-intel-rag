"""Review drafted questions against their source text, one at a time.

    uv run python eval/review.py --reviewer yourname

For each candidate you see the source excerpt, the question and the drafted answer:
  a  accept as is          e  edit the answer, then accept
  q  edit the question     r  reject
  s  skip for now          x  save and exit
Accepted questions move to questions.jsonl with verified=true and your name; only those
count in Ragas scores. Progress is saved after every decision.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

EVAL_DIR = Path(__file__).parent
CANDIDATES = EVAL_DIR / "candidates.jsonl"
QUESTIONS = EVAL_DIR / "questions.jsonl"

FETCH_CHUNK = "MATCH (c:Chunk {id: $id}) RETURN c.text AS text"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def next_question_id(questions: list[dict]) -> str:
    nums = [int(q["id"][1:]) for q in questions if q["id"][:1] == "q" and q["id"][1:].isdigit()]
    return f"q{(max(nums) if nums else 0) + 1:03d}"


def show(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.fill(p, width) for p in text.split("\n") if p.strip())


def main() -> None:
    from company_intel.db import make_driver, run_read

    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()

    candidates = read_jsonl(CANDIDATES)
    pending = [c for c in candidates if c.get("status") is None]
    print(f"{len(pending)} candidates to review.\n")
    driver = make_driver()
    try:
        for cand in pending:
            rows = run_read(driver, FETCH_CHUNK, id=cand["reference_contexts"][0])
            source = rows[0]["text"] if rows else "(source chunk not found in Neo4j)"
            print("=" * 100)
            print(f"SOURCE  {cand['reference_contexts'][0]}\n{show(source)}\n")
            print(f"QUESTION  {cand['question']}")
            print(f"ANSWER    {cand['reference_answer']}\n")
            while True:
                choice = input("[a]ccept [e]dit answer [q] edit question [r]eject [s]kip e[x]it: ")
                choice = choice.strip().lower()
                if choice in {"a", "e", "q", "r", "s", "x"}:
                    break
            if choice == "x":
                break
            if choice == "s":
                continue
            if choice == "r":
                cand["status"] = "rejected"
            else:
                old_answer = cand["reference_answer"]
                if choice == "e":
                    cand["reference_answer"] = input("New answer: ").strip() or old_answer
                if choice == "q":
                    cand["question"] = input("New question: ").strip() or cand["question"]
                    if input("Edit the answer too? [y/N] ").strip().lower() == "y":
                        cand["reference_answer"] = input("New answer: ").strip() or old_answer
                cand["status"] = "accepted"
                questions = read_jsonl(QUESTIONS)
                accepted = {k: v for k, v in cand.items() if k != "status"}
                accepted |= {
                    "id": next_question_id(questions),
                    "verified": True,
                    "verified_by": args.reviewer,
                    "edited": choice in {"e", "q"},
                }
                questions.append(accepted)
                write_jsonl(QUESTIONS, questions)
            write_jsonl(CANDIDATES, candidates)
    finally:
        driver.close()
    write_jsonl(CANDIDATES, candidates)
    n_ok = sum(1 for q in read_jsonl(QUESTIONS) if q.get("verified"))
    print(f"\n{n_ok} verified questions in {QUESTIONS.name}.")


if __name__ == "__main__":
    main()
