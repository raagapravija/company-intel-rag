"""Run the agent over verified questions and score it with Ragas.

    uv run python eval/run_ragas.py                    # all verified questions
    uv run python eval/run_ragas.py --limit 10         # quick check
    uv run python eval/run_ragas.py --metrics faithfulness context_recall

Retrieval is also scored without any LLM, against each question's gold chunk ids
(reference_contexts): retrieval_hit (gold among the hybrid candidates), rerank_hit (gold
in the reranked top-k) and mrr (reciprocal rank of the first gold chunk after reranking).

Free-tier budget: each question costs ~1 answer call plus judge calls. The default
metrics (faithfulness, factual_correctness) take ~3-4 judge calls per question; context
precision alone makes one call per retrieved context. Judge responses are cached on disk
(eval/.cache), so re-running unchanged questions is free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
ALL_METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "factual_correctness",
]
# Fits a ~100 requests/day judge quota for ~20 questions
DEFAULT_METRICS = ["faithfulness", "factual_correctness"]


def load_questions(path: Path = EVAL_DIR / "questions.jsonl") -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ready_questions(questions: list[dict]) -> list[dict]:
    """Only hand-verified questions with a reference answer count toward scores."""
    return [q for q in questions if q.get("verified") and q.get("reference_answer")]


def router_accuracy(chosen: list[str], expected: list[str]) -> float:
    """Jaccard overlap between the retrievers the router chose and the expected ones."""
    a, b = set(chosen), set(expected)
    return len(a & b) / len(a | b) if a | b else 1.0


def retrieval_scores(gold: list[str], candidates: list[str], reranked: list[str]) -> dict:
    """LLM-free retrieval metrics against the gold chunk ids; empty if there is no gold."""
    if not gold:
        return {}
    gold_set = set(gold)
    rank = next((i for i, cid in enumerate(reranked, 1) if cid in gold_set), None)
    return {
        "retrieval_hit": float(bool(gold_set & set(candidates))),
        "rerank_hit": float(rank is not None),
        "mrr": 1.0 / rank if rank else 0.0,
    }


def run_agent(questions: list[dict]) -> list[dict]:
    from company_intel.agent.graph import build_graph

    graph = build_graph()
    records = []
    for i, q in enumerate(questions, 1):
        t0 = time.time()
        state = graph.invoke({"question": q["question"]})
        contexts = [c.text for c in state.get("reranked", [])]
        records.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "reference": q["reference_answer"],
                "answer": state.get("answer", ""),
                "contexts": contexts,
                "context_ids": [c.metadata.get("chunk_id") for c in state.get("reranked", [])],
                "candidate_ids": [c.metadata.get("chunk_id") for c in state.get("contexts", [])],
                "reference_contexts": q.get("reference_contexts", []),
                "retrievers": state.get("retrievers", []),
                "expected_retrievers": q.get("expected_retrievers", []),
                "latency_s": round(time.time() - t0, 2),
            }
        )
        print(f"  [{i}/{len(questions)}] {q['id']} answered in {records[-1]['latency_s']}s")
    return records


def build_scorers(metrics: list[str]):
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics import collections as m

    from company_intel.config import get_settings

    s = get_settings()
    key = (s.eval_judge_api_key or s.llm_api_key).get_secret_value() or "not-needed"
    client = AsyncOpenAI(base_url=s.eval_judge_base_url, api_key=key)
    llm = llm_factory(
        s.eval_judge_model,
        client=client,
        cache=DiskCacheBackend(cache_dir=str(EVAL_DIR / ".cache")),
        temperature=0.0,
        max_tokens=2048,
    )
    scorers = {}
    if "context_precision" in metrics:
        scorers["context_precision"] = (
            m.ContextPrecision(llm=llm),
            lambda r: dict(user_input=r["question"], reference=r["reference"],
                           retrieved_contexts=r["contexts"]),
        )
    if "context_recall" in metrics:
        scorers["context_recall"] = (
            m.ContextRecall(llm=llm),
            lambda r: dict(user_input=r["question"], retrieved_contexts=r["contexts"],
                           reference=r["reference"]),
        )
    if "faithfulness" in metrics:
        scorers["faithfulness"] = (
            m.Faithfulness(llm=llm),
            lambda r: dict(user_input=r["question"], response=r["answer"],
                           retrieved_contexts=r["contexts"]),
        )
    if "answer_relevancy" in metrics:
        emb = HuggingFaceEmbeddings(model=s.embedding_model)
        scorers["answer_relevancy"] = (
            m.AnswerRelevancy(llm=llm, embeddings=emb),
            lambda r: dict(user_input=r["question"], response=r["answer"]),
        )
    if "factual_correctness" in metrics:
        scorers["factual_correctness"] = (
            m.FactualCorrectness(llm=llm),
            lambda r: dict(response=r["answer"], reference=r["reference"]),
        )
    return scorers


async def score(records: list[dict], metrics: list[str]) -> None:
    scorers = build_scorers(metrics)
    for i, r in enumerate(records, 1):
        r["scores"] = {
            "router_accuracy": router_accuracy(r["retrievers"], r["expected_retrievers"])
        } | retrieval_scores(r["reference_contexts"], r["candidate_ids"], r["context_ids"])
        for name, (scorer, args) in scorers.items():
            if not r["contexts"] and name != "factual_correctness":
                r["scores"][name] = 0.0
                continue
            try:
                result = await scorer.ascore(**args(r))
                r["scores"][name] = float(result.value)
            except Exception as e:  # one failed judge call shouldn't sink the run
                r["scores"][name] = None
                r.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"[:300]
        print(f"  [{i}/{len(records)}] {r['id']} scored")


def summarize(records: list[dict]) -> dict:
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        for name, val in r["scores"].items():
            if val is not None:
                by_type[r["type"]][name].append(val)
                by_type["ALL"][name].append(val)
    return {
        t: {name: round(statistics.mean(vals), 3) for name, vals in metrics.items()}
        | {"n": sum(1 for r in records if t in ("ALL", r["type"]))}
        for t, metrics in by_type.items()
    }


def print_table(summary: dict) -> None:
    metric_names = sorted({k for row in summary.values() for k in row if k != "n"})
    header = f"{'type':<12}{'n':>4}  " + "  ".join(f"{n[:18]:>18}" for n in metric_names)
    print(header)
    print("-" * len(header))
    for t in sorted(summary, key=lambda x: (x == "ALL", x)):
        row = summary[t]
        vals = "  ".join(
            f"{row[n]:>18.3f}" if n in row else f"{'-':>18}" for n in metric_names
        )
        print(f"{t:<12}{row['n']:>4}  {vals}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS, choices=ALL_METRICS)
    parser.add_argument("--label", default="", help="tag for this run, e.g. 'phase1-baseline'")
    args = parser.parse_args()

    questions = load_questions()
    ready = ready_questions(questions)
    print(f"{len(ready)}/{len(questions)} questions are verified with reference answers.")
    if not ready:
        print("Verify some questions first: python eval/review.py")
        sys.exit(1)
    if args.limit:
        ready = ready[: args.limit]

    print("Running agent...")
    records = run_agent(ready)
    print("Scoring with Ragas...")
    asyncio.run(score(records, args.metrics))

    summary = summarize(records)
    print()
    print_table(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}{'-' + args.label if args.label else ''}.json"
    from company_intel.config import get_settings

    s = get_settings()
    out.write_text(
        json.dumps(
            {
                "label": args.label,
                "config": {
                    "llm_model": s.llm_model,
                    "judge_model": s.eval_judge_model,
                    "embedding_model": s.embedding_model,
                    "rerank_model": s.rerank_model if s.rerank_provider != "none" else None,
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
