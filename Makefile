.PHONY: install up down schema test lint ingest ask draft review eval

install:
	uv sync --all-extras

up:
	docker compose up -d

down:
	docker compose down

schema:
	uv run python -m company_intel.schema

test:
	uv run pytest

lint:
	uv run ruff check src tests eval

# make ingest            all companies in config/companies.yaml
# make ingest T="NVDA AMD"
ingest:
	uv run python -m company_intel.ingestion.run $(if $(T),--tickers $(T),)

# make ask Q="What does NVIDIA say about supply-chain risk?"
ask:
	uv run python -m company_intel.ask "$(Q)"

draft:
	uv run python eval/draft_questions.py --per-company 2

# make review R=yourname
review:
	uv run python eval/review.py --reviewer $(R)

# make eval L=phase1-baseline
eval:
	uv run python eval/run_ragas.py $(if $(L),--label $(L),)
