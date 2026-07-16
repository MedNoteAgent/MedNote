"""Parse the ICD-10-CM Index XML into a synonym dictionary (Step 5.3).

The Index is a human-curated reverse lookup ("Ear infection" -> H66.9). We
flatten each nested ``<mainTerm>``/``<term>`` path into a natural-language
phrase and attach the phrases to their codes, giving SapBERT explicit synonym
signal (e.g. "heart attack" lands on I21.x).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from mednote.rag.etl.parser import ICD10Code

_INDEX_ROOT_TAG = "ICD10CM.index"
_DEFAULT_MAX_SYNONYMS = 10


def parse_icd10_index(
    xml_path: str | Path, max_synonyms: int = _DEFAULT_MAX_SYNONYMS
) -> dict[str, list[str]]:
    """Build ``{code: [natural-language phrases]}`` from the Index XML.

    Each phrase is the comma-joined trail of titles from the mainTerm down to
    the term that carries the code (e.g. "Diabetes, diabetic, with, amyotrophy").

    Raises:
        FileNotFoundError: if ``xml_path`` does not exist.
        ValueError: if the file is not an ICD-10-CM Index document or
            ``max_synonyms`` is not positive.
    """
    if max_synonyms < 1:
        raise ValueError(f"max_synonyms must be >= 1, got {max_synonyms}")
    path = Path(xml_path)
    if not path.is_file():
        raise FileNotFoundError(f"ICD-10-CM index XML not found: {path}")

    root = ET.parse(path).getroot()
    if root.tag != _INDEX_ROOT_TAG:
        raise ValueError(
            f"Expected root <{_INDEX_ROOT_TAG}> but found <{root.tag}> in {path}"
        )

    mapping: dict[str, list[str]] = {}

    def recurse(term: ET.Element, trail: list[str]) -> None:
        title = (term.findtext("title") or "").strip()
        phrase = ", ".join(t for t in trail + [title] if t)
        code = term.findtext("code")
        if code:
            bucket = mapping.setdefault(code.strip(), [])
            if phrase and phrase not in bucket and len(bucket) < max_synonyms:
                bucket.append(phrase)
        for sub in term.findall("term"):
            recurse(sub, trail + [title])

    for letter in root.findall("letter"):
        for main_term in letter.findall("mainTerm"):
            recurse(main_term, [])
    return mapping


def enrich_codes_with_synonyms(
    codes: list[ICD10Code], synonyms: dict[str, list[str]]
) -> list[ICD10Code]:
    """Merge Index phrases into ``index_synonyms``; returns NEW code objects.

    Codes without an Index entry are passed through unchanged. Inputs are never
    mutated (ICD10Code is frozen); synonym lists are copied, not shared.
    """
    return [
        replace(code, index_synonyms=list(synonyms[code.code]))
        if code.code in synonyms
        else code
        for code in codes
    ]
