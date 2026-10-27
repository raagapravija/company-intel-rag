"""Fetch 10-K / 10-Q filings from SEC EDGAR (free, no API key).

- ticker -> CIK via https://www.sec.gov/files/company_tickers.json
- filing list via https://data.sec.gov/submissions/CIK##########.json
- primary document HTML from the EDGAR archive, cached under data/raw/<TICKER>/
SEC fair-access rules: <= 10 requests/second and a User-Agent with contact info.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import requests

from company_intel.config import get_settings

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"

_MIN_INTERVAL = 0.15  # seconds between requests, stays well under 10/s
_last_request = 0.0


@dataclass(frozen=True)
class FilingRef:
    cik: str  # zero-padded to 10 digits
    ticker: str
    company_name: str
    form: str
    accession: str  # with dashes, e.g. 0001045810-25-000023
    filed_date: date
    period_of_report: date
    fiscal_year: int
    primary_doc_url: str


@lru_cache
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {"User-Agent": get_settings().sec_user_agent, "Accept-Encoding": "gzip, deflate"}
    )
    return s


def _get(url: str) -> requests.Response:
    global _last_request
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(4):
        _last_request = time.monotonic()
        resp = _session().get(url, timeout=60)
        if resp.status_code in (429, 503):
            time.sleep(2 ** (attempt + 1))
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def _cache_json(path: Path, url: str) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    data = _get(url).json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def resolve_cik(ticker: str) -> tuple[str, str]:
    """Return (10-digit CIK, company title) for a ticker."""
    data = _cache_json(get_settings().raw_dir / "company_tickers.json", TICKERS_URL)
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10), row["title"]
    raise KeyError(f"Ticker {ticker} not found in SEC ticker list")


def parse_submissions(
    data: dict, ticker: str, forms: list[str], years: list[int]
) -> list[FilingRef]:
    """Pick matching filings out of a submissions JSON (pure function, unit-tested)."""
    cik = str(data["cik"]).zfill(10)
    recent = data["filings"]["recent"]
    out: list[FilingRef] = []
    for i, form in enumerate(recent["form"]):
        if form not in forms:
            continue
        report = recent["reportDate"][i]
        if not report:
            continue
        period = date.fromisoformat(report)
        fiscal_year = period.year
        if years and fiscal_year not in years:
            continue
        accession = recent["accessionNumber"][i]
        out.append(
            FilingRef(
                cik=cik,
                ticker=ticker.upper(),
                company_name=data.get("name", ticker),
                form=form,
                accession=accession,
                filed_date=date.fromisoformat(recent["filingDate"][i]),
                period_of_report=period,
                fiscal_year=fiscal_year,
                primary_doc_url=ARCHIVE_URL.format(
                    cik_int=int(cik),
                    acc_nodash=accession.replace("-", ""),
                    doc=recent["primaryDocument"][i],
                ),
            )
        )
    return sorted(out, key=lambda f: f.period_of_report, reverse=True)


def list_filings(ticker: str, forms: list[str], years: list[int]) -> list[FilingRef]:
    cik, _ = resolve_cik(ticker)
    path = get_settings().raw_dir / ticker.upper() / "submissions.json"
    data = _cache_json(path, SUBMISSIONS_URL.format(cik=cik))
    return parse_submissions(data, ticker, forms, years)


def download_filing(ref: FilingRef) -> str:
    """Return the filing's HTML, downloading once and caching under data/raw/."""
    path = get_settings().raw_dir / ref.ticker / f"{ref.accession}.htm"
    if path.exists():
        return path.read_text(errors="ignore")
    html = _get(ref.primary_doc_url).text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return html
