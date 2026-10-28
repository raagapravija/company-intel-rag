"""Turn a 10-K's HTML into Items (sections), then into overlapping chunks.

Chunking by section first (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A...)
keeps every chunk tagged with where it came from, which sharpens retrieval and lets
Text2Cypher filter by section.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Inline-XBRL 10-Ks are XHTML; parsing them as HTML is intended and works fine.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Items we ingest from a 10-K, with display titles.
TENK_ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
}

# Any 10-K item heading at the start of a line: "Item 1A.", "ITEM 7 -", "Item 9C:" ...
# All items are matched (not only the ones we keep) so each section ends where the next begins.
ITEM_HEADING = re.compile(
    r"^[ \t]*item[ \t]*(1[0-6]|[1-9])([a-c])?\b[ \t]*[.:\-–—]?",
    re.IGNORECASE | re.MULTILINE,
)

MIN_SECTION_CHARS = 200
WORDS_PER_TOKEN = 0.75  # rough English average, good enough for sizing chunks


@dataclass
class Section:
    id: str  # f"{accession}:{item}"
    item: str
    title: str
    text: str


@dataclass
class Chunk:
    id: str  # f"{section_id}:{position}"
    section_id: str
    position: int
    text: str
    metadata: dict = field(default_factory=dict)


def html_to_text(html: str) -> str:
    """Visible text with one block element per line. Drops hidden inline-XBRL data."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    # Inline XBRL filings carry a hidden header block of machine-readable facts.
    for tag in soup.find_all(lambda t: t.name and t.name.lower() in {"ix:header"}):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = text.replace("\xa0", " ").replace("​", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    # Page numbers and lone punctuation are noise.
    lines = [ln for ln in lines if ln and not re.fullmatch(r"\d{1,3}|[-–—|]+", ln)]
    return _join_heading_lines("\n".join(lines))


def _join_heading_lines(text: str) -> str:
    """Rejoin headings split across elements, e.g. 'Item 1A.' / 'Risk Factors'."""
    return re.sub(
        r"^(item[ \t]*(?:1[0-6]|[1-9])[a-c]?\.?)\n", r"\1 ", text, flags=re.IGNORECASE | re.M
    )


def split_sections(text: str, accession: str) -> list[Section]:
    """Split a 10-K into Items. Uses "Item N." headings; if those don't yield real Risk
    Factors and MD&A sections (e.g. Intel, whose body uses plain titles and lists Items only
    in a cross-reference index), falls back to finding sections by title."""
    sections = split_sections_by_item(text, accession)
    if _looks_complete(sections):
        return sections
    by_title = split_sections_by_title(text, accession)
    return by_title if _looks_complete(by_title) else sections


# A real Risk Factors / MD&A section is pages long; a cross-reference line is not.
MIN_CORE_CHARS = {"1A": 5000, "7": 5000}


def _looks_complete(sections: list[Section]) -> bool:
    sizes = {s.item: len(s.text) for s in sections}
    return all(sizes.get(item, 0) >= n for item, n in MIN_CORE_CHARS.items())


# Plain-title headings for the fallback. Any line equal to one of these (ignoring case,
# punctuation and an optional "Item N." prefix) starts a section.
TITLE_ALIASES: dict[str, list[str]] = {
    "1": [
        "business", "our business", "fundamentals of our business", "fundamental of our business"
    ],
    "1A": ["risk factors"],
    "1C": ["cybersecurity"],
    "2": ["properties"],
    "3": ["legal proceedings"],
    "7": [
        "management's discussion and analysis",
        "management's discussion and analysis of financial condition and results of operations",
    ],
    "7A": ["quantitative and qualitative disclosures about market risk"],
}
# Other top-level titles that end a section (not ingested themselves).
STOP_TITLES = [
    "unresolved staff comments", "mine safety disclosures", "other key information",
    "market for our common stock", "critical accounting estimates", "controls and procedures",
    "financial statements and supplemental details", "financial statements and supplementary data",
    "exhibits", "exhibits and financial statement schedules", "form 10-k summary",
    "information about our executive officers", "other information",
    "disclosure pursuant to section 13(r) of the securities exchange act of 1934",
    "forward-looking statements",
]
MAX_TITLE_SECTION_CHARS = 150_000


def _normalize_title(line: str) -> str:
    line = line.strip().lower().replace("’", "'")
    line = re.sub(r"^item\s*\d{1,2}[a-c]?\s*[.:\-–—]?\s*", "", line)
    return line.rstrip(" .:")


def split_sections_by_title(text: str, accession: str) -> list[Section]:
    title_to_item = {t: item for item, ts in TITLE_ALIASES.items() for t in ts}
    stops = set(STOP_TITLES)

    # (offset, key) for every heading line; key is the item, or "STOP".
    heads: list[tuple[int, str]] = []
    pos = 0
    for line in text.split("\n"):
        norm = _normalize_title(line)
        if norm in title_to_item:
            heads.append((pos, title_to_item[norm]))
        elif norm in stops:
            heads.append((pos, "STOP"))
        pos += len(line) + 1

    best: dict[str, tuple[int, int]] = {}
    for i, (start, key) in enumerate(heads):
        if key == "STOP":
            continue
        # A section runs to the next *different* heading, so a title repeated as a page
        # header ("Risk Factors" at the top of every page) doesn't cut it short.
        end = len(text)
        for nxt, nkey in heads[i + 1 :]:
            if nkey != key:
                end = nxt
                break
        end = min(end, start + MAX_TITLE_SECTION_CHARS)
        if key not in best or (end - start) > (best[key][1] - best[key][0]):
            best[key] = (start, end)

    sections = []
    for item, title in TENK_ITEMS.items():
        if item in best:
            start, end = best[item]
            body = text[start:end].strip()
            if len(body) >= MIN_SECTION_CHARS:
                sections.append(Section(f"{accession}:{item}", item, title, body))
    return sections


def split_sections_by_item(text: str, accession: str) -> list[Section]:
    """Find each Item's body. The table of contents also lists every Item, so for each
    Item we keep the occurrence followed by the most text, which is the real section."""
    matches = list(ITEM_HEADING.finditer(text))
    best: dict[str, tuple[int, int]] = {}
    for i, m in enumerate(matches):
        item = (m.group(1) + (m.group(2) or "")).upper()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        if item not in best or (end - start) > (best[item][1] - best[item][0]):
            best[item] = (start, end)

    sections = []
    for item, title in TENK_ITEMS.items():
        if item not in best:
            continue
        start, end = best[item]
        body = text[start:end].strip()
        if len(body) < MIN_SECTION_CHARS:
            continue
        sections.append(Section(id=f"{accession}:{item}", item=item, title=title, text=body))
    return sections


def chunk_section(section: Section, max_tokens: int = 512, overlap_tokens: int = 64) -> list[Chunk]:
    """Pack paragraphs into ~max_tokens chunks; consecutive chunks share overlap_tokens."""
    max_words = int(max_tokens * WORDS_PER_TOKEN)
    overlap_words = int(overlap_tokens * WORDS_PER_TOKEN)

    # Paragraphs longer than a chunk are pre-split into word windows.
    pieces: list[list[str]] = []
    for para in section.text.split("\n"):
        words = para.split()
        for i in range(0, len(words), max_words):
            if words[i : i + max_words]:
                pieces.append(words[i : i + max_words])

    chunks_words: list[list[str]] = []
    current: list[str] = []
    for piece in pieces:
        if current and len(current) + len(piece) > max_words:
            chunks_words.append(current)
            current = current[-overlap_words:] if overlap_words else []
        current = current + ["\n"] + piece if current else list(piece)
    if current:
        chunks_words.append(current)

    chunks = []
    for pos, words in enumerate(chunks_words):
        text = " ".join(words).replace(" \n ", "\n")
        chunks.append(
            Chunk(
                id=f"{section.id}:{pos}",
                section_id=section.id,
                position=pos,
                text=text,
            )
        )
    return chunks
