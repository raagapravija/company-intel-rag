# CLAUDE.md

Context for Claude Code working in this repo. Read README.md, docs/schema.md and
eval/README.md for the full design; this file covers what isn't obvious from them.

## What this is

Hybrid RAG over a Neo4j knowledge graph of SEC 10-K filings for 12 semiconductor
companies (config/companies.yaml, FY2024 + FY2025). A LangGraph agent routes a question
to retrievers, reranks, answers with numbered citations, and is scored with Ragas.
Portfolio project; the goal is to show measured gains of hybrid/graph retrieval over a
plain-vector baseline, per question type.

## Hard constraints

- **$0 budget, no credit card.** Embeddings and reranking run locally; the LLM is Groq's
  free tier. Don't add paid APIs. Voyage is an optional extra only.
- **Never commit `.env`** (it holds the Groq key and Neo4j password). It is git-ignored.
- **Reference answers are never invented.** Only questions a person verified against the
  source (`verified: true` in eval/questions.jsonl) count toward scores. Don't write or
  "fix" reference answers from model knowledge; numeric references must come from SEC data.

## Stack and current settings

| Piece | Choice |
|---|---|
| DB | Neo4j 5.26 Community in Docker (`docker-compose.yml`), APOC enabled |
| Embeddings | `BAAI/bge-m3`, local, 1024-dim, normalized (`src/company_intel/embeddings.py`) |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder, local; top 30 -> 8 |
| Search | Neo4j vector index (HNSW, cosine) + full-text (Lucene BM25), fused with RRF k=60 |
| LLM | Groq `openai/gpt-oss-120b`, `reasoning_effort=low`, via OpenAI-compatible client (`llm.py`) |
| Ragas judge | Gemini `gemini-3.8-flash` via AI Studio free tier (different family from the answering model; ~100 req/day) |
| Orchestration | LangGraph: route -> retrieve -> rerank -> generate -> verify (retry loop) |

Groq free tier: ~30 requests/min, ~8,000 tokens/min, ~1,000 requests/day per model.
`llm.complete()` honors `retry-after`. Groq retired the Llama models on 2026-08-16; check
https://console.groq.com/docs/deprecations before changing models.

## Commands

Run from the repo root on macOS (uv manages `.venv`; conda's `(base)` doesn't interfere).

```bash
docker compose up -d --wait   # start Neo4j and wait until it accepts connections
make schema                   # constraints + indexes (idempotent)
make test                     # unit tests, no Neo4j/models/keys needed
make lint                     # ruff
make ingest T="NVDA AMD"      # ingest; skips filings already loaded with the same model
uv run python -m company_intel.ingestion.run --tickers INTC --force   # reload a company
make ask Q="What does NVIDIA say about supply-chain risk?"
make draft                    # LLM drafts candidate questions -> eval/candidates.jsonl
make review R=pravija         # interactive human review (needs a real terminal)
make eval-retrieval L=phase1-baseline  # LLM-free retrieval scores, primary eval, no API calls
make eval L=phase1-baseline   # Ragas (judge quota-limited; use --limit for spot checks)
```

Neo4j Browser: http://localhost:7474 (user `neo4j`, password from `.env`).

## Layout

```
src/company_intel/
  config.py           Settings from .env; companies.yaml loader
  db.py               make_driver, run_read/run_write (cypher + params are positional-only)
  schema.py           constraints, vector + full-text indexes
  embeddings.py, llm.py
  ingestion/          edgar.py (fetch+cache), chunking.py, load.py, run.py (CLI)
                      structured.py, extract.py, news.py are stubs for phases 2-4
  retrieval/          hybrid.py, rerank.py; text2cypher.py, graph.py are stubs
  agent/              state.py, graph.py (LangGraph), answer.py (prompt + citations)
  ask.py, api/main.py CLI and FastAPI
eval/                 questions.jsonl, candidates.jsonl, draft_questions.py, review.py, run_ragas.py
tests/                pytest; everything external is mocked
```

## Gotchas learned the hard way

- **10-K section splitting** (`ingestion/chunking.py`): uses "Item N." headings, keeping
  the longest occurrence per Item to skip the table of contents. Intel's 10-K has no Item
  headings in the body (only a cross-reference index), so `split_sections` falls back to
  `split_sections_by_title`, which also ignores titles repeated as page headers. After any
  change, check every filing with the Cypher below: each should have 1A and 7 with many chunks.
- **Re-ingesting** a filing needs `--force`; `load_filing` deletes the filing's old
  sections/chunks first so nothing stale remains.
- **Company/year filters**: Neo4j 5's vector index can't pre-filter, and post-filtering left
  as few as 11 of 30 results. When a ticker or fiscal year is named, `hybrid_search` scores
  exact cosine over just those chunks (`RETRIEVAL_EXACT_FILTERED`). The year comes from
  "fiscal YYYY 10-K" in the question (`RETRIEVAL_YEAR_FILTER`); setting both false
  reproduces the phase 1 baseline.
- **Groq's `qwen/qwen3.8-27b` can't be the judge**: 1,000 output tokens/min cap rejects every
  Ragas call (429). Judge moved to Gemini; default metrics are faithfulness +
  factual_correctness plus LLM-free retrieval_hit / rerank_hit / mrr on gold chunk ids.
- **Reranker needs the source header**: the cross-encoder used to see bare chunk text (no
  company/year/section), so table-heavy MD&A chunks lost to look-alikes. Prepending the
  same header used at embedding (`RERANK_WITH_HEADER`) took rerank MRR 0.72 -> 0.93.
  Collapsing whitespace (`RERANK_COMPACT_TEXT`) had no effect. Reranking costs ~40-65 s
  per question on CPU.
- **Ragas 0.4** needs `langchain-community<0.4` (pinned). Uses `ragas.metrics.collections`
  and `llm_factory(..., client=AsyncOpenAI(...))`. Judge calls are cached in `eval/.cache`.
- **gpt-oss is a reasoning model**: reasoning tokens count against `max_tokens`
  (2048); an empty answer with `finish_reason=length` raises a clear error.
- Docker Desktop's virtual disk filled up once ("No space left on device"); fix with
  `docker system prune -a` and a bigger disk limit in Docker Desktop settings.

Section health check (Neo4j Browser):

```cypher
MATCH (c:Company)-[:FILED]->(f:Filing)
OPTIONAL MATCH (f)-[:HAS_SECTION]->(s:Section)-[:HAS_CHUNK]->(ch)
WITH c.ticker AS ticker, f.fiscal_year AS fy, s.item AS item, count(ch) AS chunks
ORDER BY ticker, fy, item
RETURN ticker, fy, collect(item + ':' + chunks) AS sections
```

## Status (2026-10-28)

- Phase 0 skeleton and Phase 1 pipeline done; all 24 filings ingested and healthy.
- `make ask` works end to end.
- Eval: 20 lookup questions verified; remaining candidates unreviewed.
  q001-q007 are unverified seed examples for later phases.
- Primary eval is LLM-free (`eval/run_retrieval.py`): hit@k/MRR/recall@30 on gold chunk ids,
  text_hit@8 (same paragraph elsewhere), same_filing@8 (year bias), across vector /
  fulltext / hybrid / hybrid+rerank. Ragas is optional: free judges are too rate-limited
  for full runs. Gold ids depend on chunking, so keep chunking fixed while comparing.

First answer observations (to measure, not yet fix): answers favor the older FY2024
filing over FY2025 and don't state the year per fact; NVIDIA's dependence on TSMC/foundries
was missing from the supply-chain answer.

## Next steps

1. Review ~11 more candidates (to ~20 verified), then `make eval L=phase1-baseline`.
2. Only after the baseline: retrieval improvements, each re-scored with Ragas
   (exact pre-filtered cosine when a company is named, since each has ~250 chunks;
   prefer the latest filing unless a year is asked; prompt for per-fact years).
3. Phase 2: XBRL financial facts + 13F holdings, Text2Cypher with a read-only Neo4j user,
   numeric questions whose references come from executed Cypher.
4. Phase 3: LLM entity/relationship extraction into the graph, graph-traversal retriever,
   relational questions. Phase 4: news. Phase 5: router, LLM grounding verifier, Langfuse, UI.

## Working conventions

- Run `make test` and `make lint` before committing; add a test for every bug fixed.
- Keep commits small with clear messages. Push to `origin main`
  (github.com/raagapravija/company-intel-rag).
- When changing retrieval or prompts, report Ragas scores per question type before vs after.
