from company_intel.config import Settings, load_companies


def test_company_list_loads():
    cfg = load_companies()
    assert cfg.sector
    assert len(cfg.companies) >= 5
    tickers = [c.ticker for c in cfg.companies]
    assert len(tickers) == len(set(tickers)), "duplicate tickers"


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.embedding_dimensions > 0
    assert s.neo4j_uri.startswith(("bolt://", "neo4j://", "neo4j+s://", "bolt+s://"))
