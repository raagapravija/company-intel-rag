from company_intel.retrieval.hybrid import reciprocal_rank_fusion
from company_intel.retrieval.text2cypher import is_read_only


def test_rrf_rewards_items_ranked_by_both():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]])
    ids = [i for i, _ in fused]
    assert ids[0] in {"a", "b"}
    assert set(ids) == {"a", "b", "c", "d"}
    assert ids.index("d") > ids.index("b")


def test_read_only_guard():
    assert is_read_only("MATCH (c:Company) WHERE c.ticker = 'NVDA' RETURN c.name")
    assert is_read_only("MATCH (c:Company) RETURN c SKIP 5 LIMIT 10")
    assert not is_read_only("MATCH (c:Company) DETACH DELETE c")
    assert not is_read_only("MERGE (c:Company {cik: '1'})")
    assert not is_read_only("MATCH (c) SET c.x = 1")
    assert not is_read_only("CALL apoc.create.node(['X'], {})")


def test_agent_graph_compiles():
    from company_intel.agent.graph import build_graph

    assert build_graph() is not None


def test_hybrid_search_modes(monkeypatch):
    import company_intel.db as db
    import company_intel.embeddings as emb
    from company_intel.retrieval import hybrid

    def row(i):
        return {"id": i, "text": i, "score": 1.0, "ticker": "X"}

    calls = []

    def fake_read(driver, cypher, **params):
        calls.append("vector" if "vector" in cypher else "fulltext")
        return [row("v1"), row("v2")] if calls[-1] == "vector" else [row("f1"), row("v2")]

    monkeypatch.setattr(db, "make_driver", lambda s: type("D", (), {"close": lambda self: None})())
    monkeypatch.setattr(db, "run_read", fake_read)
    monkeypatch.setattr(emb, "embed_query", lambda q, s: [0.0])

    def ids(mode):
        calls.clear()
        out = hybrid.hybrid_search("q", k=3, tickers=["X"], mode=mode)
        return [c.metadata["chunk_id"] for c in out], list(calls)

    assert ids("vector") == (["v1", "v2"], ["vector"])
    assert ids("fulltext") == (["f1", "v2"], ["fulltext"])
    fused, used = ids("hybrid")
    assert fused[0] == "v2" and set(fused) == {"v1", "v2", "f1"} and used == ["vector", "fulltext"]
