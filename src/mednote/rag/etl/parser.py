"""Parse the ICD-10-CM Tabular XML into self-contained code documents (Step 5.2).

Critical insight (docs/implementation_plan.md Task 5): standard word-count
chunking would destroy this data. Each ``<diag>`` element becomes exactly one
:class:`ICD10Code` document, enriched with the full ancestor hierarchy so it
stands alone at embedding time.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_TABULAR_ROOT_TAG = "ICD10CM.tabular"


@dataclass(frozen=True)
class ICD10Code:
    """A single ICD-10-CM code as a self-contained document for embedding."""

    code: str                                   # e.g. "E11.9"
    description: str                            # from <desc>
    hierarchy_path: str                         # chapter -> section -> ancestors
    chapter: str
    chapter_code: str
    includes: list[str] = field(default_factory=list)
    inclusion_terms: list[str] = field(default_factory=list)
    excludes1: list[str] = field(default_factory=list)
    excludes2: list[str] = field(default_factory=list)
    code_first: list[str] = field(default_factory=list)
    use_additional_code: list[str] = field(default_factory=list)
    parent_code: str | None = None
    children_codes: list[str] = field(default_factory=list)
    index_synonyms: list[str] = field(default_factory=list)  # Step 5.3
    target_sex: list[str] = field(default_factory=list)      # Step 5.4
    max_age_days: int | None = None                          # Step 5.4

    def to_embedding_text(self) -> str:
        """Fuse code + description + hierarchy + every synonym source."""
        parts = [
            f"{self.code}: {self.description}",
            f"Hierarchy: {self.hierarchy_path}",
        ]
        synonyms = self.includes + self.inclusion_terms + self.index_synonyms
        if synonyms:
            parts.append("Also known as: " + ", ".join(synonyms))
        if self.excludes1:
            parts.append("Excludes: " + ", ".join(self.excludes1[:5]))
        return "\n".join(parts)


def _notes(diag: ET.Element, tag: str) -> list[str]:
    """Collect <note> texts under a child element such as <includes>."""
    element = diag.find(tag)
    if element is None:
        return []
    return [n.text.strip() for n in element.findall("note") if n.text and n.text.strip()]


def _walk_diag(
    diag: ET.Element,
    hierarchy_parts: list[str],
    chapter: str,
    chapter_code: str,
    parent_code: str | None,
    out: list[ICD10Code],
) -> None:
    """Depth-first walk; each <diag> becomes one document, children recurse."""
    code = diag.findtext("name", "").strip()
    description = diag.findtext("desc", "").strip()
    child_elements = diag.findall("diag")

    out.append(
        ICD10Code(
            code=code,
            description=description,
            hierarchy_path=" -> ".join(p for p in hierarchy_parts if p),
            chapter=chapter,
            chapter_code=chapter_code,
            includes=_notes(diag, "includes"),
            inclusion_terms=_notes(diag, "inclusionTerm"),
            excludes1=_notes(diag, "excludes1"),
            excludes2=_notes(diag, "excludes2"),
            code_first=_notes(diag, "codeFirst"),
            use_additional_code=_notes(diag, "useAdditionalCode"),
            parent_code=parent_code,
            children_codes=[c.findtext("name", "").strip() for c in child_elements],
        )
    )
    for child in child_elements:
        # Child hierarchy extends with THIS diag's description.
        _walk_diag(child, hierarchy_parts + [description], chapter, chapter_code, code, out)


def parse_icd10_tabular(xml_path: str | Path) -> list[ICD10Code]:
    """Parse the CMS Tabular XML into a flat list of ICD10Code documents.

    Raises:
        FileNotFoundError: if ``xml_path`` does not exist.
        ValueError: if the file is not an ICD-10-CM Tabular document.
    """
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"ICD-10-CM tabular XML not found: {path}")

    root = ET.parse(path).getroot()
    if root.tag != _TABULAR_ROOT_TAG:
        raise ValueError(
            f"Expected root <{_TABULAR_ROOT_TAG}> but found <{root.tag}> in {path}"
        )

    codes: list[ICD10Code] = []
    for chapter in root.findall("chapter"):
        chapter_code = chapter.findtext("name", "").strip()
        chapter_desc = chapter.findtext("desc", "").strip()
        for section in chapter.findall("section"):
            section_desc = section.findtext("desc", "").strip()
            for diag in section.findall("diag"):
                _walk_diag(
                    diag,
                    [chapter_desc, section_desc],
                    chapter=chapter_desc,
                    chapter_code=chapter_code,
                    parent_code=None,
                    out=codes,
                )
    return codes
