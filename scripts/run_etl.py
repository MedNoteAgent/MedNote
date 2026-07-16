"""One-command ICD-10-CM ETL: raw CMS XML -> enriched JSONL (Task 5).

Steps (docs/implementation_plan.md Step 5.5):
    1. parse_icd10_tabular          tabular XML -> ICD10Code documents
    2. parse_icd10_index            index XML -> {code: [synonyms]}
    3. enrich_codes_with_synonyms   merge synonyms into codes
    4. apply_metadata_tags          sex/age hard-filter tags
    5. export_to_jsonl              write the JSONL hand-off artifact

Usage (from the repo root):
    uv run python scripts/run_etl.py
"""

from __future__ import annotations

import sys
import time

from mednote.config import get_config
from mednote.rag.etl import (
    apply_metadata_tags,
    enrich_codes_with_synonyms,
    export_to_jsonl,
    parse_icd10_index,
    parse_icd10_tabular,
)


def main() -> int:
    cfg = get_config()
    started = time.perf_counter()

    print(f"[1/5] Parsing tabular XML: {cfg.paths.icd10_tabular_path}")
    codes = parse_icd10_tabular(cfg.paths.icd10_tabular_path)
    print(f"      -> {len(codes):,} ICD-10-CM code documents")

    print(f"[2/5] Parsing index XML:   {cfg.paths.icd10_index_path}")
    synonyms = parse_icd10_index(
        cfg.paths.icd10_index_path, max_synonyms=cfg.etl.max_index_synonyms
    )
    print(f"      -> synonym phrases for {len(synonyms):,} distinct codes")

    print("[3/5] Enriching codes with index synonyms")
    enriched = enrich_codes_with_synonyms(codes, synonyms)
    enriched_count = sum(1 for c in enriched if c.index_synonyms)
    print(f"      -> {enriched_count:,} of {len(enriched):,} codes enriched")

    print("[4/5] Applying sex/age metadata tags")
    tagged = apply_metadata_tags(enriched)
    female = sum(1 for c in tagged if c.target_sex == ["female"])
    male = sum(1 for c in tagged if c.target_sex == ["male"])
    perinatal = sum(1 for c in tagged if c.max_age_days is not None)
    print(f"      -> female-only: {female:,} | male-only: {male} | perinatal: {perinatal}")

    print(f"[5/5] Exporting JSONL:     {cfg.paths.icd10_processed_path}")
    out_path = export_to_jsonl(tagged, cfg.paths.icd10_processed_path)
    size_mb = out_path.stat().st_size / 1e6
    elapsed = time.perf_counter() - started
    print(f"      -> {len(tagged):,} lines, {size_mb:.1f} MB in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ETL failed: {exc}", file=sys.stderr)
        print(
            "Hint: run `uv run python scripts/download_icd10.py` first if the "
            "source XML is missing.",
            file=sys.stderr,
        )
        sys.exit(1)
