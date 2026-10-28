# Company Intel RAG

Hybrid retrieval-augmented generation over a Neo4j knowledge graph of SEC filings,
structured financials, institutional holdings and news. A LangGraph agent routes each
question to the right retrieval method (or several), reranks the results, answers with
citations and checks that every claim is grounded. Ragas measures each part.

## Why hybrid

| Question | Plain vector RAG | This system |
|---|---|---|
| "What does NVIDIA say about supply-chain risk?" | works | hybrid search |
| "Which companies had revenue above $20B?" | fuzzy guess | Text2Cypher on XBRL facts |
| "Which suppliers do AMD and NVIDIA share?" | misses connections | graph traversal |
| "Which investors hold both AMAT and LRCX, and what export risks do those companies disclose?" | fails | all three, combined |

## Architecture

**Ingestion**

```
10-K / 10-Q / news ─┬─> chunk + embed (BGE-M3, local) ───> vector + full-text index ─┐
                    └─> LLM entity extraction ──────────> entities + relationships ─┼─> Neo4j
XBRL / Form 13F ──────> structured load ────────────────> financial facts, holdings ┘
```

**Query (LangGraph)**

```
question -> router -> { hybrid search | Text2Cypher | graph traversal } -> rerank
         -> generate + verify -> answer with sources
              ^                          |
              └──── retry if ungrounded ─┘
```

Everything lives in one Neo4j database, so a single query can go from a vector hit to the
company it belongs to and on to that company's suppliers. See [docs/schema.md](docs/schema.md).

## Stack

| Layer | Choice |
|---|---|
| Graph + vector + full-text | Neo4j 5 (Community, Docker) |
| Embeddings / reranking | `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3`, run locally (Voyage optional) |
| Orchestration | LangGraph |
| LLM | Groq free tier (`openai/gpt-oss-120b`) or Ollama, via one OpenAI-compatible client |
| Evaluation | Ragas (judge: `qwen/qwen3.8-27b` on Groq) + custom router metric |
| Tracing | Langfuse (free tier) |
| API / UI | FastAPI, Streamlit |

**Cost: $0, no credit card.** SEC EDGAR is free, embeddings and reranking run on your
machine, and Groq's free tier covers answering and evaluation (about 1,000 requests/day
per model). For fully offline use, point `LLM_BASE_URL` at Ollama (see `.env.example`).

## Setup

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and Docker. About 6 GB of disk for
PyTorch and the two local models (downloaded on first use).

```bash
cp .env.example .env      # set NEO4J_PASSWORD, LLM_API_KEY (Groq), SEC_USER_AGENT (your email)
make install              # uv sync --all-extras
make up                   # start Neo4j (browser at http://localhost:7474)
make schema               # create constraints and indexes
make test                 # unit tests (no Neo4j or API keys needed)
```

Then:

```bash
make ingest T="NVDA AMD"  # try two companies first; `make ingest` does all 12
make ask Q="What does NVIDIA say about supply-chain risk?"
```

Ingestion caches filings under `data/raw/` and skips anything already loaded, so it is
safe to stop and re-run. On a laptop CPU, embedding one 10-K takes a few minutes.

Companies to ingest are listed in [config/companies.yaml](config/companies.yaml)
(12 semiconductor companies; one sector so the graph is densely connected).

## Layout

```
src/company_intel/
  config.py, db.py, schema.py   settings, Neo4j driver, constraints + indexes
  embeddings.py, llm.py         local/Voyage embeddings; OpenAI-compatible LLM client
  ingestion/                    edgar, chunking, load, run (phase 1); structured, extract, news
  retrieval/                    hybrid, rerank (phase 1); text2cypher, graph
  agent/                        LangGraph state + graph, answer generation with citations
  ask.py, api/                  CLI and FastAPI app
eval/                           questions, draft -> review workflow, Ragas runner
docs/schema.md                  graph model
```

## Roadmap

Each phase ends with a Ragas run compared against the phase 1 baseline.

- [x] **Phase 0** Skeleton: config, Docker, schema, agent graph shape, tests
- [x] **Phase 1** Plain vector RAG: EDGAR fetch, section-aware chunking, local embeddings, hybrid search, rerank, cited answers, question review workflow, Ragas runner
- [ ] **Phase 1b** First real run: ingest, verify ~15 lookup questions, record baseline Ragas scores
- [ ] **Phase 2** Structured data: XBRL financial facts, 13F holdings, Text2Cypher with read-only user
- [ ] **Phase 3** Knowledge graph: LLM entity + relationship extraction, entity resolution, graph traversal retriever
- [ ] **Phase 4** News ingestion and recency-aware retrieval
- [ ] **Phase 5** LangGraph router + grounding verifier with retry, Langfuse tracing, UI

## Evaluation

See [eval/README.md](eval/README.md). Every reference answer is checked by a person
against the source filing before it counts; numeric references come straight from SEC
data. Scores are reported per question type so each retriever's contribution is visible.
