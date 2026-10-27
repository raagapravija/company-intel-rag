"""Fetch 10-K / 10-Q filings from SEC EDGAR.  [Phase 1]

Plan:
- Resolve ticker -> CIK via https://www.sec.gov/files/company_tickers.json
- List filings via https://data.sec.gov/submissions/CIK##########.json
- Download the primary document HTML for each matching form/year into data/raw/
- SEC asks for <= 10 requests/second and a User-Agent with contact info (SEC_USER_AGENT)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class FilingRef:
    cik: str
    ticker: str
    company_name: str
    form: str
    accession: str
    filed_date: date
    fiscal_year: int
    primary_doc_url: str


def resolve_cik(ticker: str) -> str:
    raise NotImplementedError("Phase 1")


def list_filings(cik: str, forms: list[str], years: list[int]) -> list[FilingRef]:
    raise NotImplementedError("Phase 1")


def download_filing(ref: FilingRef) -> str:
    """Download and return the filing's HTML (cached under data/raw/)."""
    raise NotImplementedError("Phase 1")
