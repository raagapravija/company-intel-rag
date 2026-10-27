# Evaluation

Only questions a person has checked against the filings count toward scores. Nothing is
scored from memory or from unreviewed LLM output.

## How reference answers are made

| Type | Target | Where the reference answer comes from |
|---|---|---|
| `lookup` | ~15 | `make draft` drafts a question + answer from one real chunk; you verify it against that chunk in `make review R=yourname` |
| `numeric` | ~10 | Phase 2: the result of a hand-checked Cypher query over SEC XBRL data (no LLM involved) |
| `relational` | ~10 | Phase 3: candidate links from the graph, each confirmed by finding it in both filings |
| `summary` | ~5 | written last; the reference lists the themes an answer must cover |

Workflow for lookup questions (after `make ingest`):

```bash
make draft              # ~2 candidates per company -> eval/candidates.jsonl (unverified)
make review R=yourname  # read the source excerpt, then accept / edit / reject each one
make eval L=phase1-baseline
```

`review` shows the exact source chunk next to each drafted answer, so checking takes a
minute or two per question. Accepted ones are appended to `questions.jsonl` with
`verified: true` and your name.

The seven seed questions in `questions.jsonl` (q001-q007) are examples of what to cover per
type; they stay `verified: false` until someone writes and checks their answers.

## Question fields

| field | meaning |
|---|---|
| `id` | stable id, never reused |
| `type` | `lookup`, `numeric`, `relational` or `summary`; scores are reported per type |
| `question` | as a real analyst would ask it |
| `reference_answer` | ground truth, checked by a person against the source |
| `reference_contexts` | chunk ids that contain the answer |
| `reference_cypher` | numeric questions only: a correct query, used for result-set comparison |
| `expected_retrievers` | what the router should pick (router accuracy) |
| `verified`, `verified_by`, `edited` | review status |

## Metrics

| Part of the system | Metric |
|---|---|
| Retrieval | Ragas context precision, context recall |
| Answers | Ragas faithfulness, answer relevancy, factual correctness |
| Router | chosen retrievers vs `expected_retrievers` |
| Text2Cypher (phase 2) | execute generated vs reference Cypher, compare result sets |

The judge is `EVAL_JUDGE_MODEL` (default `openai/gpt-oss-120b` on Groq's free tier), a
different model from the one answering. Judge calls are cached in `eval/.cache`, so
re-running unchanged questions uses no quota. Budget roughly 10-15 judge calls per
question; on Groq's free tier (about 1,000 requests/day per model) that is one full run of
50 questions per day, or use `--metrics` / `--limit` for smaller runs.

Results are saved to `eval/results/<timestamp>-<label>.json` (git-ignored) with the model
config, per-type summary and per-question details.
