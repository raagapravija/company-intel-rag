from company_intel.ingestion.chunking import (
    Section,
    chunk_section,
    html_to_text,
    split_sections,
)

FILLER = "The company designs accelerated computing platforms. " * 40

SAMPLE_10K = f"""
<html><head><title>10-K</title><style>p {{color: red}}</style></head><body>
<div style="display:none"><ix:header>hidden xbrl facts 12345</ix:header></div>
<p>TABLE OF CONTENTS</p>
<table>
<tr><td>Item 1.</td><td>Business</td><td>4</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>12</td></tr>
<tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>30</td></tr>
<tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>40</td></tr>
<tr><td>Item 8.</td><td>Financial Statements</td><td>55</td></tr>
</table>
<p>PART I</p>
<p><b>Item 1.</b></p><p><b>Business</b></p>
<p>{FILLER}</p>
<p>See Item 1A for risks. This mention is mid-paragraph and must not split the section.</p>
<p>4</p>
<p><b>Item 1A. Risk Factors</b></p>
<p>We depend on third parties to manufacture our products. {FILLER}</p>
<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>
<p>PART II</p>
<p>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</p>
<p>Revenue grew strongly. {FILLER}</p>
<p>Item 8. Financial Statements</p><p>{FILLER}</p>
</body></html>
"""


def test_html_to_text_drops_hidden_and_page_numbers():
    text = html_to_text(SAMPLE_10K)
    assert "hidden xbrl facts" not in text
    assert "color: red" not in text
    assert "\n4\n" not in text
    # Heading split across two elements is rejoined
    assert "Item 1. Business" in text


def test_split_sections_skips_table_of_contents():
    sections = {s.item: s for s in split_sections(html_to_text(SAMPLE_10K), "0000-25-1")}
    assert set(sections) == {"1", "1A", "7"}  # 1B/8 aren't kept; 1C, 2, 3, 7A are absent
    assert "accelerated computing" in sections["1"].text
    assert "mid-paragraph" in sections["1"].text
    assert sections["1A"].text.startswith("Item 1A")
    assert "third parties" in sections["1A"].text
    assert "Unresolved" not in sections["1A"].text
    assert "Revenue grew" in sections["7"].text
    assert "Financial Statements" not in sections["7"].text
    assert sections["1A"].id == "0000-25-1:1A"


def test_chunks_are_bounded_overlapping_and_ordered():
    words = [f"w{i}" for i in range(2000)]
    paras = [" ".join(words[i : i + 50]) for i in range(0, 2000, 50)]
    section = Section(id="acc:1A", item="1A", title="Risk Factors", text="\n".join(paras))
    chunks = chunk_section(section, max_tokens=200, overlap_tokens=20)

    assert len(chunks) > 5
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.id == f"acc:1A:{c.position}" for c in chunks)
    for c in chunks:
        assert len(c.text.split()) <= 150 + 15 + 5
    # Consecutive chunks share words (overlap) and together cover every word
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert set(a.text.split()) & set(b.text.split())
    covered = set(" ".join(c.text for c in chunks).split())
    assert covered == set(words)
