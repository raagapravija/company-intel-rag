# Evaluation

`questions.jsonl` is the test set. Each line:

| field | meaning |
|---|---|
| `id` | stable id, never reused |
| `type` | `lookup`, `numeric`, `relational` or `summary`; scores are reported per type |
| `question` | as a real analyst would ask it |
| `reference_answer` | ground truth, written by hand after checking the filings |
| `reference_contexts` | chunk ids that contain the answer (for context recall) |
| `reference_cypher` | numeric questions only: a correct query, used for result-set comparison |
| `expected_retrievers` | what the router should pick (router accuracy) |

The seven seed questions have no reference answers yet. Fill them in from the actual
filings once ingestion runs; don't generate them with an LLM. Target 30-50 hand-written
questions, then extend with Ragas synthetic generation.

## Metrics

| Part of the system | Metric |
|---|---|
| Retrieval | Ragas context precision, context recall |
| Answers | Ragas faithfulness, answer relevancy, factual correctness |
| Text2Cypher | custom: execute generated vs reference Cypher, compare result sets |
| Router | custom: chosen retrievers vs `expected_retrievers` |

Run: `make eval`. Results land in `eval/results/` (git-ignored), one file per run, so each
phase can be compared with the plain-vector baseline.
