from __future__ import annotations

from typing import Iterable, List

from bs4 import BeautifulSoup, Tag

from .text_clean import html_to_text


def _split_sections_by_headings(html: str) -> List[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    if not headings:
        return []

    sections: List[str] = []
    for idx, heading in enumerate(headings):
        texts: List[str] = [heading.get_text(" ", strip=True)]
        for sib in heading.next_siblings:
            if isinstance(sib, Tag) and sib.name and sib.name.lower() in {
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            }:
                break
            texts.append(sib.get_text(" ", strip=True) if isinstance(sib, Tag) else str(sib))
        section_html = " ".join(part for part in texts if part)
        sections.append(section_html)
    return sections


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start == end:
            break
    return [c for c in chunks if c]


def chunk_html(html: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    sections = _split_sections_by_headings(html)
    text_sections: List[str] = []
    if sections:
        for section in sections:
            text_sections.append(html_to_text(section))
    else:
        text_sections = [html_to_text(html)]

    chunks: List[str] = []
    for section in text_sections:
        chunks.extend(_chunk_text(section, chunk_size, chunk_overlap))
    return chunks
