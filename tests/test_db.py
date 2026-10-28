from company_intel.db import run_read, run_write


class FakeRecord:
    def __init__(self, d):
        self._d = d

    def data(self):
        return self._d


class FakeDriver:
    def __init__(self):
        self.calls = []

    def execute_query(self, cypher, params, **kwargs):
        self.calls.append((cypher, params, kwargs))
        return [FakeRecord({"ok": 1})], None, None


def test_params_may_be_named_query_or_driver():
    d = FakeDriver()
    rows = run_read(d, "RETURN $query, $driver", query="nvidia", driver="x", k=5)
    assert rows == [{"ok": 1}]
    cypher, params, kwargs = d.calls[0]
    assert cypher == "RETURN $query, $driver"
    assert params == {"query": "nvidia", "driver": "x", "k": 5}
    assert "routing_" in kwargs

    run_write(d, "MERGE (n {q: $query})", query="q")
    assert d.calls[1][1] == {"query": "q"}
