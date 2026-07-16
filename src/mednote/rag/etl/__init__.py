"""ICD-10-CM ETL pipeline: XML -> enriched, self-contained code documents.

Public API (one function per pipeline step, orchestrated by scripts/run_etl.py):

    parse_icd10_tabular         Step 5.2 — Tabular XML -> ICD10Code documents
    parse_icd10_index           Step 5.3 — Index XML -> {code: [synonyms]}
    enrich_codes_with_synonyms  Step 5.3 — merge synonyms into codes
    apply_metadata_tags         Step 5.4 — sex/age hard-filter tags
    export_to_jsonl             Step 5.5 — write JSONL hand-off artifact
"""

from mednote.rag.etl.export import export_to_jsonl
from mednote.rag.etl.index_parser import enrich_codes_with_synonyms, parse_icd10_index
from mednote.rag.etl.metadata import AGE_RESTRICTIONS, SEX_RESTRICTIONS, apply_metadata_tags
from mednote.rag.etl.parser import ICD10Code, parse_icd10_tabular

__all__ = [
    "AGE_RESTRICTIONS",
    "ICD10Code",
    "SEX_RESTRICTIONS",
    "apply_metadata_tags",
    "enrich_codes_with_synonyms",
    "export_to_jsonl",
    "parse_icd10_index",
    "parse_icd10_tabular",
]
