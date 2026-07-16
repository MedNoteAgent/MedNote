"""Tests for the ICD-10-CM ETL pipeline (Task 5).

Unit tests run against small inline XML fixtures that mirror the real CMS
structure. The integration test at the bottom runs against the actual 2026
corpus in data/corpus/ and asserts the counts validated in
docs/notebooks/icd10_rag.ipynb; it is skipped when the XML is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mednote.rag.etl import (
    ICD10Code,
    apply_metadata_tags,
    enrich_codes_with_synonyms,
    export_to_jsonl,
    parse_icd10_index,
    parse_icd10_tabular,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TABULAR = REPO_ROOT / "data" / "corpus" / "icd10cm_tabular_2026.xml"
REAL_INDEX = REPO_ROOT / "data" / "corpus" / "icd10cm_index_2026.xml"

TABULAR_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<ICD10CM.tabular>
  <version>2026</version>
  <chapter>
    <name>4</name>
    <desc>Endocrine, nutritional and metabolic diseases (E00-E89)</desc>
    <section id="E08-E13">
      <desc>Diabetes mellitus (E08-E13)</desc>
      <diag>
        <name>E11</name>
        <desc>Type 2 diabetes mellitus</desc>
        <includes>
          <note>diabetes NOS</note>
          <note>insulin resistant diabetes (mellitus)</note>
        </includes>
        <useAdditionalCode>
          <note>code to identify control using insulin (Z79.4)</note>
        </useAdditionalCode>
        <excludes1>
          <note>gestational diabetes (O24.4-)</note>
        </excludes1>
        <diag>
          <name>E11.9</name>
          <desc>Type 2 diabetes mellitus without complications</desc>
          <inclusionTerm>
            <note>Type 2 diabetes mellitus NOS</note>
          </inclusionTerm>
        </diag>
        <diag>
          <name>E11.4</name>
          <desc>Type 2 diabetes mellitus with neurological complications</desc>
          <diag>
            <name>E11.44</name>
            <desc>Type 2 diabetes mellitus with diabetic amyotrophy</desc>
          </diag>
        </diag>
      </diag>
    </section>
  </chapter>
</ICD10CM.tabular>
"""

INDEX_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<ICD10CM.index>
  <version>2026</version>
  <title>ICD-10-CM INDEX TO DISEASES and INJURIES</title>
  <letter>
    <title>D</title>
    <mainTerm>
      <title>Diabetes, diabetic</title>
      <code>E11.9</code>
      <term level="1">
        <title>with</title>
        <term level="2">
          <title>amyotrophy</title>
          <code>E11.44</code>
        </term>
      </term>
    </mainTerm>
    <mainTerm>
      <title>Dwarfism</title>
      <code>E34.328</code>
    </mainTerm>
  </letter>
</ICD10CM.index>
"""


@pytest.fixture()
def tabular_path(tmp_path: Path) -> Path:
    path = tmp_path / "tabular.xml"
    path.write_text(TABULAR_FIXTURE, encoding="utf-8")
    return path


@pytest.fixture()
def index_path(tmp_path: Path) -> Path:
    path = tmp_path / "index.xml"
    path.write_text(INDEX_FIXTURE, encoding="utf-8")
    return path


@pytest.fixture()
def parsed_codes(tabular_path: Path) -> list[ICD10Code]:
    return parse_icd10_tabular(tabular_path)


# ---------------------------------------------------------------- tabular ---


def test_parse_tabular_emits_one_document_per_diag(parsed_codes: list[ICD10Code]) -> None:
    assert [c.code for c in parsed_codes] == ["E11", "E11.9", "E11.4", "E11.44"]


def test_parse_tabular_builds_hierarchy_and_links(parsed_codes: list[ICD10Code]) -> None:
    by_code = {c.code: c for c in parsed_codes}

    e11 = by_code["E11"]
    assert e11.parent_code is None
    assert e11.children_codes == ["E11.9", "E11.4"]
    assert e11.hierarchy_path == (
        "Endocrine, nutritional and metabolic diseases (E00-E89)"
        " -> Diabetes mellitus (E08-E13)"
    )
    assert e11.chapter_code == "4"

    e1144 = by_code["E11.44"]
    assert e1144.parent_code == "E11.4"
    assert e1144.children_codes == []
    # Hierarchy accumulates ancestor descriptions on the way down.
    assert e1144.hierarchy_path.endswith(
        "Type 2 diabetes mellitus -> Type 2 diabetes mellitus with neurological complications"
    )


def test_parse_tabular_collects_note_elements(parsed_codes: list[ICD10Code]) -> None:
    by_code = {c.code: c for c in parsed_codes}

    assert by_code["E11"].includes == [
        "diabetes NOS",
        "insulin resistant diabetes (mellitus)",
    ]
    assert by_code["E11"].excludes1 == ["gestational diabetes (O24.4-)"]
    assert by_code["E11"].use_additional_code == [
        "code to identify control using insulin (Z79.4)"
    ]
    assert by_code["E11.9"].inclusion_terms == ["Type 2 diabetes mellitus NOS"]


def test_parse_tabular_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_icd10_tabular(tmp_path / "nope.xml")


def test_parse_tabular_rejects_wrong_root(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("<wrong/>", encoding="utf-8")
    with pytest.raises(ValueError, match="ICD10CM.tabular"):
        parse_icd10_tabular(bad)


def test_to_embedding_text_fuses_all_synonym_sources() -> None:
    code = ICD10Code(
        code="G44.2",
        description="Tension-type headache",
        hierarchy_path="Nervous system -> Episodic disorders",
        chapter="Nervous system",
        chapter_code="6",
        includes=["tension headache NOS"],
        inclusion_terms=["stress headache"],
        index_synonyms=["Headache, tension"],
        excludes1=["headache NOS (R51.9)"],
    )
    text = code.to_embedding_text()
    assert text.splitlines()[0] == "G44.2: Tension-type headache"
    assert "Hierarchy: Nervous system -> Episodic disorders" in text
    assert "Also known as: tension headache NOS, stress headache, Headache, tension" in text
    assert "Excludes: headache NOS (R51.9)" in text


# ------------------------------------------------------------------ index ---


def test_parse_index_flattens_term_paths(index_path: Path) -> None:
    mapping = parse_icd10_index(index_path)
    assert mapping["E11.9"] == ["Diabetes, diabetic"]
    assert mapping["E11.44"] == ["Diabetes, diabetic, with, amyotrophy"]
    assert mapping["E34.328"] == ["Dwarfism"]


def test_parse_index_caps_synonyms_per_code(tmp_path: Path) -> None:
    terms = "".join(
        f'<term level="1"><title>variant {i}</title><code>A00.0</code></term>'
        for i in range(6)
    )
    xml = (
        '<ICD10CM.index><letter><title>A</title>'
        f"<mainTerm><title>Cholera</title>{terms}</mainTerm>"
        "</letter></ICD10CM.index>"
    )
    path = tmp_path / "capped.xml"
    path.write_text(xml, encoding="utf-8")

    mapping = parse_icd10_index(path, max_synonyms=3)
    assert len(mapping["A00.0"]) == 3


def test_parse_index_rejects_wrong_root(tmp_path: Path) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("<wrong/>", encoding="utf-8")
    with pytest.raises(ValueError, match="ICD10CM.index"):
        parse_icd10_index(bad)


def test_enrich_merges_synonyms_without_mutating_inputs(
    parsed_codes: list[ICD10Code], index_path: Path
) -> None:
    mapping = parse_icd10_index(index_path)
    enriched = enrich_codes_with_synonyms(parsed_codes, mapping)

    by_code = {c.code: c for c in enriched}
    assert by_code["E11.9"].index_synonyms == ["Diabetes, diabetic"]
    assert by_code["E11.44"].index_synonyms == ["Diabetes, diabetic, with, amyotrophy"]
    assert by_code["E11"].index_synonyms == []  # no index entry for the parent

    # Immutability: the originals must be untouched.
    assert all(c.index_synonyms == [] for c in parsed_codes)


# --------------------------------------------------------------- metadata ---


def _make_code(code: str) -> ICD10Code:
    return ICD10Code(
        code=code,
        description="desc",
        hierarchy_path="a -> b",
        chapter="chapter",
        chapter_code="1",
    )


def test_apply_metadata_tags_sex_and_age() -> None:
    codes = [_make_code(c) for c in ("O80", "N40.1", "P22.0", "E11.9")]
    tagged = apply_metadata_tags(codes)

    by_code = {c.code: c for c in tagged}
    assert by_code["O80"].target_sex == ["female"]
    assert by_code["N40.1"].target_sex == ["male"]
    assert by_code["P22.0"].max_age_days == 28
    assert by_code["E11.9"].target_sex == []
    assert by_code["E11.9"].max_age_days is None

    # Immutability: original objects keep their empty tags.
    assert all(c.target_sex == [] and c.max_age_days is None for c in codes)


# ----------------------------------------------------------------- export ---


def test_export_to_jsonl_round_trips(parsed_codes: list[ICD10Code], tmp_path: Path) -> None:
    out = tmp_path / "processed" / "codes.jsonl"
    written = export_to_jsonl(parsed_codes, out)
    assert written == out

    raw = out.read_bytes()
    # One record per line, terminated by REAL newlines (the notebook prototype
    # accidentally wrote literal backslash-n).
    assert raw.count(b"\n") == len(parsed_codes)
    assert b"\\n" not in raw.splitlines()[0][-4:]

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["code"] for r in records] == ["E11", "E11.9", "E11.4", "E11.44"]
    assert records[0]["children_codes"] == ["E11.9", "E11.4"]
    assert records[0]["includes"] == ["diabetes NOS", "insulin resistant diabetes (mellitus)"]


# ------------------------------------------------------------ integration ---


@pytest.mark.skipif(
    not (REAL_TABULAR.exists() and REAL_INDEX.exists()),
    reason="real CMS 2026 XML not present in data/corpus/",
)
def test_full_etl_against_real_cms_2026_corpus(tmp_path: Path) -> None:
    """End-to-end against the real corpus; numbers validated in the notebook."""
    codes = parse_icd10_tabular(REAL_TABULAR)
    assert len(codes) == 46_881

    by_code = {c.code: c for c in codes}
    g442 = by_code["G44.2"]
    assert g442.description == "Tension-type headache"
    assert g442.parent_code == "G44"
    assert g442.children_codes == ["G44.20", "G44.21", "G44.22"]
    assert g442.hierarchy_path == (
        "Diseases of the nervous system (G00-G99)"
        " -> Episodic and paroxysmal disorders (G40-G47)"
        " -> Other headache syndromes"
    )

    mapping = parse_icd10_index(REAL_INDEX)
    assert len(mapping) == 20_347

    enriched = enrich_codes_with_synonyms(codes, mapping)
    enriched_count = sum(1 for c in enriched if c.index_synonyms)
    assert enriched_count == 16_390
    i219 = next(c for c in enriched if c.code == "I21.9")
    assert any("Infarct" in s for s in i219.index_synonyms)

    tagged = apply_metadata_tags(enriched)
    assert sum(1 for c in tagged if c.target_sex == ["female"]) == 1_791
    assert sum(1 for c in tagged if c.target_sex == ["male"]) == 27
    assert sum(1 for c in tagged if c.max_age_days is not None) == 565

    out = tmp_path / "icd10_codes.jsonl"
    export_to_jsonl(tagged, out)
    assert sum(1 for _ in out.open(encoding="utf-8")) == 46_881
