"""Load structured data: XBRL financial facts and Form 13F holdings.  [Phase 2]

- XBRL: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
  -> (:Company)-[:REPORTED]->(:FinancialFact {concept, value, unit, period_end, fiscal_year})
- 13F: institutional holdings
  -> (:Institution)-[:HOLDS {shares, value_usd, quarter}]->(:Company)
"""

from __future__ import annotations


def load_financial_facts(cik: str, concepts: list[str] | None = None) -> int:
    raise NotImplementedError("Phase 2")


def load_13f_holdings(quarter: str) -> int:
    raise NotImplementedError("Phase 2")
