"""Split a 10-K into Items, then into overlapping chunks.  [Phase 1]

Chunking by section first (Item 1 Business, Item 1A Risk Factors, Item 7 MD&A...)
keeps every chunk tagged with where it came from, which sharpens retrieval and lets
Text2Cypher filter by section.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    raise NotImplementedError("Phase 1")


def split_sections(text: str, accession: str) -> list[Section]:
    raise NotImplementedError("Phase 1")


def chunk_section(section: Section, max_tokens: int = 512, overlap: int = 64) -> list[Chunk]:
    raise NotImplementedError("Phase 1")
