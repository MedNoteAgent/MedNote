"""Export processed ICD-10 codes to JSONL (Step 5.5).

One JSON object per line — the hand-off artifact between the ETL and the
indexing job (scripts/build_index.py).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mednote.rag.etl.parser import ICD10Code


def export_to_jsonl(codes: list[ICD10Code], output_path: str | Path) -> Path:
    """Write one JSON object per line; creates parent directories as needed.

    Raises:
        ValueError: if ``codes`` is empty — an empty knowledge base is always
            an upstream bug, never a valid export.
    """
    if not codes:
        raise ValueError("Refusing to export an empty code list")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for code in codes:
            handle.write(json.dumps(asdict(code), ensure_ascii=False) + "\n")
    return path
