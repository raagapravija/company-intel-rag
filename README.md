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
10-K / 10-Q / news ─┬─> chunk + embed (voyage-4-large) ──> vector + full-text index ─┐
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
| Embeddings / reranking | Voyage `voyage-4-large`, Voyage reranker |
| Orchestration | LangGraph |
| LLM | Claude |
| Evaluation | Ragas + custom router and Text2Cypher metrics |
| Tracing | Langfuse |
| API / UI | FastAPI, Streamlit |

## Setup

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
cp .env.example .env      # add Neo4j password, VOYAGE_API_KEY, ANTHROPIC_API_KEY, SEC_USER_AGENT
make install              # uv sync --all-extras
make up                   # start Neo4j (browser at http://localhost:7474)
make schema               # create constraints and indexes
make test                 # unit tests (no Neo4j or API keys needed)
```

Companies to ingest are listed in [config/companies.yaml](config/companies.yaml)
(12 semiconductor companies to start; one sector so the graph is densely connected).

## Layout

```
src/company_intel/
  config.py, db.py, schema.py   settings, Neo4j driver, constraints + indexes
  ingestion/                    edgar, chunking, embed, structured, extract, news
  retrieval/                    hybrid, rerank, text2cypher, graph
  agent/                        LangGraph state + graph
  api/                          FastAPI app
eval/                           questions.jsonl, run_ragas.py
docs/schema.md                  graph model
```

## Roadmap

Each phase ends with a Ragas run compared against the phase 1 baseline.

- [x] **Phase 0** Skeleton: config, Docker, schema, agent graph shape, tests
- [ ] **Phase 1** Plain vector RAG: EDGAR fetch, section-aware chunking, Voyage embeddings, hybrid search, rerank, answer. Baseline Ragas scores.
- [ ] **Phase 2** Structured data: XBRL financial facts, 13F holdings, Text2Cypher with read-only user
- [ ] **Phase 3** Knowledge graph: LLM entity + relationship extraction, entity resolution, graph traversal retriever
- [ ] **Phase 4** News ingestion and recency-aware retrieval
- [ ] **Phase 5** LangGraph router + grounding verifier with retry, Langfuse tracing, UI

## Evaluation

See [eval/README.md](eval/README.md). Hand-write 30-50 questions with reference answers
across four types (lookup, numeric, relational, summary), then extend with Ragas
synthetic generation. Scores are reported per question type so each retriever's
contribution is visible.
