"""Answer generation with numbered, checkable citations."""

from __future__ import annotations

import re

from company_intel.llm import complete
from company_intel.retrieval import RetrievedContext

SYSTEM = """You answer questions about public companies using only the numbered sources \
provided, which are excerpts from SEC filings.

Rules:
- Every factual sentence ends with the source number(s) it relies on, like [1] or [2][4].
- Use only the sources. If they don't contain the answer, say what is missing instead \
of guessing.
- When sources cover different fiscal years, say which year each fact is from and prefer \
the most recent unless asked otherwise.
- Be concise: a short direct answer first, then supporting detail."""


def source_label(ctx: RetrievedContext) -> str:
    m = ctx.metadata
    if ctx.source == "hybrid":
        return (
            f"{m.get('ticker')} {m.get('form')} FY{m.get('fiscal_year')}, "
            f"Item {m.get('item')} {m.get('section')}"
        )
    return m.get("label", ctx.source)


def format_sources(contexts: list[RetrievedContext]) -> str:
    return "\n\n".join(
        f"[{i}] {source_label(c)}\n{c.text}" for i, c in enumerate(contexts, start=1)
    )


def cited_numbers(answer: str, n_sources: int) -> list[int]:
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    return sorted(n for n in nums if 1 <= n <= n_sources)


def generate_answer(question: str, contexts: list[RetrievedContext]) -> tuple[str, list[dict]]:
    if not contexts:
        return "I couldn't find anything relevant in the ingested filings.", []
    user = f"Sources:\n\n{format_sources(contexts)}\n\nQuestion: {question}"
    answer = complete(SYSTEM, user)
    citations = []
    for n in cited_numbers(answer, len(contexts)):
        c = contexts[n - 1]
        citations.append({"n": n, "label": source_label(c), **c.metadata})
    return answer, citations
