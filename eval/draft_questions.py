"""Draft candidate lookup questions from ingested chunks, for a human to verify.

    uv run python eval/draft_questions.py --per-company 2

Picks random chunks from Risk Factors, MD&A and Business, asks the LLM for one
question answerable from that chunk alone plus its answer, and appends them to
eval/candidates.jsonl as unverified. Nothing here counts toward scores until it passes
review (eval/review.py), where you check each answer against the source text.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

EVAL_DIR = Path(__file__).parent
CANDIDATES = EVAL_DIR / "candidates.jsonl"

SAMPLE_CHUNKS = """
MATCH (co:Company {ticker: $ticker})-[:FILED]->(f:Filing)-[:HAS_SECTION]->(s:Section)
      -[:HAS_CHUNK]->(c:Chunk)
WHERE s.item IN $items AND size(c.text) > 800
RETURN c.id AS chunk_id, c.text AS text, co.ticker AS ticker, co.name AS company,
       f.fiscal_year AS fiscal_year, s.item AS item, s.title AS section
"""

SYSTEM = """You write evaluation questions for a retrieval system over SEC 10-K filings.
Given one excerpt, write ONE question an analyst might ask that this excerpt fully answers,
and the answer, using only facts stated in the excerpt.

Rules:
- Name the company and fiscal year in the question (e.g. "In its fiscal 2025 10-K, ...").
- The question must not quote the excerpt or hint at its wording.
- The answer is 1-3 sentences, specific, and fully supported by the excerpt.
- If the excerpt is boilerplate or has no specific facts, return {"skip": true}.

Return JSON: {"question": "...", "answer": "..."} or {"skip": true}."""


def existing_chunk_ids() -> set[str]:
    ids: set[str] = set()
    for path in (CANDIDATES, EVAL_DIR / "questions.jsonl"):
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    ids.update(json.loads(line).get("reference_contexts", []))
    return ids


def main() -> None:
    from company_intel.config import load_companies
    from company_intel.db import make_driver, run_read
    from company_intel.llm import complete

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-company", type=int, default=2)
    parser.add_argument("--items", nargs="*", default=["1A", "7", "1"])
    parser.add_argument("--tickers", nargs="*")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    used = existing_chunk_ids()
    companies = load_companies().companies
    if args.tickers:
        companies = [c for c in companies if c.ticker in {t.upper() for t in args.tickers}]

    driver = make_driver()
    n_written = 0
    try:
        with open(CANDIDATES, "a") as out:
            for co in companies:
                rows = [
                    r for r in run_read(driver, SAMPLE_CHUNKS, ticker=co.ticker, items=args.items)
                    if r["chunk_id"] not in used
                ]
                rng.shuffle(rows)
                made = 0
                for r in rows:
                    if made >= args.per_company:
                        break
                    user = (
                        f"Company: {r['company']} ({r['ticker']}), 10-K fiscal {r['fiscal_year']}, "
                        f"Item {r['item']} {r['section']}\n\nExcerpt:\n{r['text']}"
                    )
                    try:
                        draft = json.loads(complete(SYSTEM, user, json_mode=True))
                    except json.JSONDecodeError:
                        continue
                    if draft.get("skip") or not draft.get("question"):
                        continue
                    made += 1
                    n_written += 1
                    record = {
                        "id": f"cand-{r['chunk_id']}",
                        "type": "lookup",
                        "question": draft["question"],
                        "reference_answer": draft["answer"],
                        "reference_contexts": [r["chunk_id"]],
                        "expected_retrievers": ["hybrid"],
                        "verified": False,
                        "drafted_by": "llm",
                    }
                    out.write(json.dumps(record) + "\n")
                    print(f"{co.ticker}: {draft['question']}")
    finally:
        driver.close()
    print(f"\nWrote {n_written} candidates to {CANDIDATES}. Review them: python eval/review.py")


if __name__ == "__main__":
    main()
