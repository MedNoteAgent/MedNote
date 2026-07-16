"""Download the ICD-10-CM 2026 XML from CMS.gov into data/corpus/ (Step 5.1).

Idempotent: exits immediately when both XML files are already present. ICD-10
updates annually every October — bump `etl.download_url` and the two
`paths.icd10_*_path` entries in config.yml for a yearly refresh.

Usage (from the repo root):
    uv run python scripts/download_icd10.py
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import httpx

from mednote.config import get_config

_DOWNLOAD_TIMEOUT_S = 120.0


def _extract_member(archive: zipfile.ZipFile, filename: str, target: Path) -> None:
    """Find `filename` anywhere in the archive tree and write it to `target`."""
    matches = [m for m in archive.namelist() if Path(m).name.lower() == filename.lower()]
    if not matches:
        raise FileNotFoundError(f"'{filename}' not found in the downloaded archive")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(archive.read(matches[0]))
    print(f"  extracted {filename} -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    cfg = get_config()
    tabular = Path(cfg.paths.icd10_tabular_path)
    index = Path(cfg.paths.icd10_index_path)

    if tabular.is_file() and index.is_file():
        print(f"Already present, nothing to do:\n  {tabular}\n  {index}")
        return 0

    print(f"Downloading {cfg.etl.download_url} ...")
    response = httpx.get(
        cfg.etl.download_url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT_S
    )
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        _extract_member(archive, tabular.name, tabular)
        _extract_member(archive, index.name, index)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (httpx.HTTPError, zipfile.BadZipFile, FileNotFoundError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "Manual fallback: fetch the '2026 Code Tables, Tabular and Index' zip "
            "from https://www.cms.gov/medicare/coding-billing/icd-10-codes and place "
            "both XML files in data/corpus/.",
            file=sys.stderr,
        )
        sys.exit(1)
