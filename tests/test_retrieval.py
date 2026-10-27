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
