.PHONY: install up down schema test lint eval

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

eval:
	uv run python eval/run_ragas.py
