"""Agent pipeline with retrieval and the LLM mocked out: no Neo4j, models or API keys."""

from company_intel.agent import answer as answer_mod
from company_intel.agent.answer import cited_numbers, format_sources
from company_intel.config import CompanyConfig, CompanyRef
from company_intel.retrieval import RetrievedContext
from company_intel.retrieval.hybrid import detect_tickers, lucene_escape

COMPANIES = CompanyConfig(
    sector="Semis",
    companies=[
        CompanyRef(ticker="NVDA", name="NVIDIA Corp", aliases=["nvidia"]),
        CompanyRef(ticker="AMD", name="Advanced Micro Devices", aliases=["amd"]),
        CompanyRef(ticker="MU", name="Micron Technology", aliases=["micron"]),
    ],
)


def ctx(i: int, ticker: str = "NVDA") -> RetrievedContext:
    return RetrievedContext(
        text=f"chunk text {i}",
        source="hybrid",
        score=1.0 / (i + 1),
        metadata={
            "chunk_id": f"acc:1A:{i}", "ticker": ticker, "form": "10-K", "fiscal_year": 2025,
            "item": "1A", "section": "Risk Factors", "url": "https://sec.gov/x",
        },
    )


def test_detect_tickers_by_alias_and_ticker():
    assert detect_tickers("What does NVIDIA say about TSMC?", COMPANIES) == ["NVDA"]
    assert detect_tickers("Compare amd and MU margins", COMPANIES) == ["AMD", "MU"]
    assert detect_tickers("Which companies face export controls?", COMPANIES) is None
    # whole words only: "mu" inside "much" is not Micron
    assert detect_tickers("how much revenue?", COMPANIES) is None


def test_lucene_escape():
    assert lucene_escape('AI/ML (data-center) "growth"?') == (
        'AI\\/ML \\(data\\-center\\) \\"growth\\"\\?'
    )


def test_citations_parsed_and_bounded():
    assert cited_numbers("A [1]. B [2][3]. C [9].", n_sources=3) == [1, 2, 3]
    assert "[2] NVDA 10-K FY2025, Item 1A Risk Factors" in format_sources([ctx(0), ctx(1)])


def test_graph_end_to_end_with_mocks(monkeypatch):
    import company_intel.retrieval.hybrid as hybrid
    import company_intel.retrieval.rerank as rerank_mod
    from company_intel.agent.graph import build_graph

    contexts = [ctx(i) for i in range(5)] + [ctx(0)]  # duplicate gets dropped
    monkeypatch.setattr(hybrid, "hybrid_search", lambda q: contexts)
    monkeypatch.setattr(rerank_mod, "rerank", lambda q, cs: cs[:3])
    monkeypatch.setattr(
        answer_mod, "complete", lambda system, user: "Supply depends on TSMC [1][3]."
    )

    result = build_graph().invoke({"question": "What does NVIDIA say about supply risk?"})
    assert result["answer"] == "Supply depends on TSMC [1][3]."
    assert [c["n"] for c in result["citations"]] == [1, 3]
    assert result["citations"][0]["chunk_id"] == "acc:1A:0"
    assert result["grounded"] is True
    assert result["attempts"] == 1
    assert len(result["reranked"]) == 3


def test_uncited_answer_is_ungrounded_but_does_not_loop(monkeypatch):
    import company_intel.retrieval.hybrid as hybrid
    import company_intel.retrieval.rerank as rerank_mod
    from company_intel.agent.graph import build_graph

    monkeypatch.setattr(hybrid, "hybrid_search", lambda q: [ctx(0)])
    monkeypatch.setattr(rerank_mod, "rerank", lambda q, cs: cs)
    monkeypatch.setattr(answer_mod, "complete", lambda system, user: "No idea.")

    result = build_graph().invoke({"question": "q"})
    assert result["grounded"] is False
    assert result["attempts"] == 1
