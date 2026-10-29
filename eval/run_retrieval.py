"""Score retrieval without any LLM, against each verified question's gold chunk ids.

    uv run python eval/run_retrieval.py --label phase1-baseline

Runs the same questions through four configs: vector (the plain-vector baseline),
fulltext, hybrid (RRF) and hybrid+rerank. Needs only Neo4j and the local models.

Metrics per question and config:
  hit@1, hit@3, hit@8, mrr   gold chunk (by id) in the top 1/3/8; reciprocal rank in the top 8
  recall@30                  gold chunk among all candidates, before reranking
  text_hit@8                 a top-8 chunk repeats most of the gold chunk's sentences, e.g. the
                             same paragraph in the other year's filing or after re-chunking
  same_filing@8              share of the top 8 from the gold chunk's filing (year bias)

Gold ids are a lower bound: identical text elsewhere only counts in text_hit@8.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_ragas import RESULTS_DIR, load_questions, ready_questions  # noqa: E402

CONFIGS = ["vector", "fulltext", "hybrid", "hybrid+rerank"]
TOP_K = 8
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def filing_of(chunk_id: str) -> str:
    """Chunk ids are '<accession>:<item>:<n>'."""
    return chunk_id.rsplit(":", 2)[0]


def sentences(text: str, min_len: int = 40) -> set[str]:
    norm = " ".join(text.split()).lower()
    return {s for s in _SENTENCE_END.split(norm) if len(s) >= min_len}


def same_text(gold: str, text: str, min_share: float = 0.5) -> bool:
    """True if `text` holds at least min_share of gold's sentences. Consecutive chunks
    overlap by ~12%, so neighbours of the gold chunk don't count."""
    g = sentences(gold)
    return bool(g) and len(g & sentences(text)) / len(g) >= min_share


def score_ranking(
    gold_ids: list[str], gold_texts: list[str], top: list[tuple[str, str]], candidates: list[str]
) -> dict:
    """top: (chunk_id, text) of the final top-k, best first; candidates: all retrieved ids."""
    gold = set(gold_ids)
    ids = [cid for cid, _ in top]
    rank = next((i for i, cid in enumerate(ids, 1) if cid in gold), None)
    filing = filing_of(gold_ids[0])
    return {
        "hit@1": float(rank is not None and rank <= 1),
        "hit@3": float(rank is not None and rank <= 3),
        f"hit@{TOP_K}": float(rank is not None),
        "mrr": 1.0 / rank if rank else 0.0,
        "recall@30": float(bool(gold & set(candidates))),
        f"text_hit@{TOP_K}": float(
            rank is not None or any(same_text(g, t) for g in gold_texts for _, t in top)
        ),
        f"same_filing@{TOP_K}": (
            sum(filing_of(cid) == filing for cid in ids) / len(ids) if ids else 0.0
        ),
    }


def load_gold(ids: list[str]) -> dict[str, dict]:
    from company_intel.db import make_driver, run_read

    driver = make_driver()
    try:
        rows = run_read(
            driver,
            """
            MATCH (f:Filing)-[:HAS_SECTION]->(:Section)-[:HAS_CHUNK]->(c:Chunk)
            WHERE c.id IN $ids
            RETURN c.id AS id, c.text AS text, f.fiscal_year AS fiscal_year
            """,
            ids=ids,
        )
    finally:
        driver.close()
    return {r["id"]: r for r in rows}


def run(questions: list[dict]) -> list[dict]:
    from company_intel.retrieval.hybrid import hybrid_search
    from company_intel.retrieval.rerank import rerank

    gold = load_gold([cid for q in questions for cid in q["reference_contexts"]])
    records = []
    for i, q in enumerate(questions, 1):
        known = [cid for cid in q["reference_contexts"] if cid in gold]
        if not known:
            print(f"  [{i}/{len(questions)}] {q['id']} skipped: gold chunk not in the graph")
            continue
        gold_texts = [gold[cid]["text"] for cid in known]
        t0 = time.time()
        found = {mode: hybrid_search(q["question"], mode=mode) for mode in CONFIGS[:3]}
        reranked = rerank(q["question"], list(found["hybrid"]), top_k=TOP_K)
        scores = {}
        for config in CONFIGS:
            cands = found["hybrid" if config == "hybrid+rerank" else config]
            top = reranked if config == "hybrid+rerank" else cands[:TOP_K]
            scores[config] = score_ranking(
                known,
                gold_texts,
                [(c.metadata["chunk_id"], c.text) for c in top],
                [c.metadata["chunk_id"] for c in cands],
            )
        records.append(
            {
                "id": q["id"],
                "type": q["type"],
                "fiscal_year": gold[known[0]]["fiscal_year"],
                "question": q["question"],
                "gold": known,
                "scores": scores,
            }
        )
        print(f"  [{i}/{len(questions)}] {q['id']} in {time.time() - t0:.1f}s")
    return records


def summarize(records: list[dict]) -> dict:
    """{group: {config: {metric: mean}}} for ALL, each question type and each fiscal year."""
    vals: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        for group in ("ALL", r["type"], f"FY{r['fiscal_year']}"):
            for config, metrics in r["scores"].items():
                for name, v in metrics.items():
                    vals[group][config][name].append(v)
    return {
        group: {
            config: {name: round(statistics.mean(v), 3) for name, v in metrics.items()}
            | {"n": len(next(iter(metrics.values())))}
            for config, metrics in configs.items()
        }
        for group, configs in vals.items()
    }


def print_table(summary: dict) -> None:
    for group in sorted(summary, key=lambda g: (g != "ALL", g)):
        configs = summary[group]
        names = [k for k in next(iter(configs.values())) if k != "n"]
        n = next(iter(configs.values()))["n"]
        print(f"\n{group} (n={n})")
        header = f"{'config':<15}" + "".join(f"{m:>15}" for m in names)
        print(header)
        print("-" * len(header))
        for config in CONFIGS:
            if config in configs:
                print(f"{config:<15}" + "".join(f"{configs[config][m]:>15.3f}" for m in names))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="", help="tag for this run, e.g. 'phase1-baseline'")
    args = parser.parse_args()

    ready = [q for q in ready_questions(load_questions()) if q.get("reference_contexts")]
    print(f"{len(ready)} verified questions have gold chunk ids.")
    if not ready:
        sys.exit(1)
    if args.limit:
        ready = ready[: args.limit]

    records = run(ready)
    summary = summarize(records)
    print_table(summary)

    from company_intel.config import get_settings

    s = get_settings()
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}{'-' + args.label if args.label else ''}-retrieval.json"
    out.write_text(
        json.dumps(
            {
                "label": args.label,
                "config": {
                    "embedding_model": s.embedding_model,
                    "rerank_model": s.rerank_model if s.rerank_provider != "none" else None,
                    "retrieval_k": s.retrieval_k,
                    "top_k": TOP_K,
                },
                "summary": summary,
                "records": records,
            },
            indent=2,
        )
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
