"""Chunk the clinical guidelines corpus by section heading (Step 6.4).

Unlike ICD-10 codes (one document per code), the guidelines are prose. Each
``##`` section in data/corpus/clinical_guidelines.md is written to be
self-contained, so a section IS the chunk — no word-count splitting that
would sever a red-flag rule from its escalation instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter

_SECTION_HEADER = "section"
_HEADERS_TO_SPLIT_ON = [("#", "title"), ("##", _SECTION_HEADER)]


@dataclass(frozen=True)
class GuidelineChunk:
    """One self-contained guideline section, ready for embedding."""

    heading: str
    text: str
    source: str  # corpus file name, kept in the payload for citation

    def to_embedding_text(self) -> str:
        return f"{self.heading}\n{self.text}"


def load_guideline_chunks(md_path: str | Path) -> list[GuidelineChunk]:
    """Split a guidelines markdown file into one chunk per ``##`` section.

    Content before the first ``##`` (the document preamble) carries no
    guideline and is dropped.

    Raises:
        FileNotFoundError: if ``md_path`` does not exist.
        ValueError: if the file contains no ``##`` sections.
    """
    path = Path(md_path)
    if not path.is_file():
        raise FileNotFoundError(f"Guidelines corpus not found: {path}")

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT_ON)
    documents = splitter.split_text(path.read_text(encoding="utf-8"))

    chunks = [
        GuidelineChunk(
            heading=doc.metadata[_SECTION_HEADER],
            text=doc.page_content.strip(),
            source=path.name,
        )
        for doc in documents
        if _SECTION_HEADER in doc.metadata
    ]
    if not chunks:
        raise ValueError(f"No '##' sections found in guidelines corpus: {path}")
    return chunks
