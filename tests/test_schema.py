from company_intel.config import Settings
from company_intel.schema import UNIQUE_KEYS, VECTOR_INDEX, schema_statements


def test_statements_are_idempotent():
    for stmt in schema_statements(Settings(_env_file=None)):
        assert "IF NOT EXISTS" in stmt


def test_vector_index_uses_configured_dimensions():
    stmts = schema_statements(Settings(_env_file=None, embedding_dimensions=2048))
    vector = [s for s in stmts if VECTOR_INDEX in s]
    assert len(vector) == 1
    assert "`vector.dimensions`: 2048" in vector[0]


def test_one_constraint_per_key():
    stmts = schema_statements(Settings(_env_file=None))
    constraints = [s for s in stmts if s.startswith("CREATE CONSTRAINT")]
    assert len(constraints) == len(UNIQUE_KEYS)
