from datetime import date

from company_intel.ingestion.edgar import parse_submissions

SUBMISSIONS = {
    "cik": "1045810",
    "name": "NVIDIA CORP",
    "filings": {
        "recent": {
            "form": ["10-Q", "10-K", "8-K", "10-K", "10-K/A", "10-K"],
            "accessionNumber": [
                "0001045810-25-000100",
                "0001045810-25-000023",
                "0001045810-25-000050",
                "0001045810-24-000029",
                "0001045810-24-000031",
                "0001045810-23-000017",
            ],
            "filingDate": [
                "2025-05-28", "2025-02-26", "2025-03-01", "2024-02-21", "2024-03-01", "2023-02-24"
            ],
            "reportDate": [
                "2025-04-27", "2025-01-26", "", "2024-01-28", "2024-01-28", "2023-01-29"
            ],
            "primaryDocument": [
                "q1.htm", "nvda-20250126.htm", "8k.htm", "nvda-20240128.htm", "a.htm", "old.htm"
            ],
        }
    },
}


def test_parse_submissions_filters_form_and_year():
    refs = parse_submissions(SUBMISSIONS, "nvda", ["10-K"], [2024, 2025])
    assert [r.accession for r in refs] == ["0001045810-25-000023", "0001045810-24-000029"]
    latest = refs[0]
    assert latest.ticker == "NVDA"
    assert latest.cik == "0001045810"
    assert latest.fiscal_year == 2025
    assert latest.period_of_report == date(2025, 1, 26)
    assert latest.primary_doc_url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
    )


def test_parse_submissions_no_year_filter_returns_all_matching():
    refs = parse_submissions(SUBMISSIONS, "NVDA", ["10-K", "10-K/A"], [])
    assert len(refs) == 4
