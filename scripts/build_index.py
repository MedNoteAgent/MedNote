"""One-command index build: ETL JSONL + guidelines -> embedded Qdrant (Task 6).

Steps (docs/implementation_plan.md Steps 6.2-6.4):
    1. Load SapBERT + the BM25 sparse encoder
    2. Open the embedded Qdrant store (creates the hybrid collection)
    3. Embed + upsert all ICD-10 code documents from the ETL JSONL
    4. Embed + upsert the clinical guidelines corpus (same collection,
       doc_type="guideline")

Single-process constraint: embedded Qdrant locks data/qdrant_data/ — let this
script exit before starting the UI, eval scripts, or validate_index.py.

Usage (from the repo root):
    uv run python scripts/build_index.py
"""

from __future__ import annotations

import sys
import time

from mednote.config import get_config
from mednote.rag.embeddings import Bm25SparseEncoder, ClinicalEmbedder
from mednote.rag.guidelines import load_guideline_chunks
from mednote.rag.indexer import ICD10Indexer, get_qdrant_client


def _print_progress(done: int, total: int) -> None:
    print(f"      ... {done:,}/{total:,} documents upserted", flush=True)


def main() -> int:
    cfg = get_config()
    started = time.perf_counter()

    print(f"[1/4] Loading encoders: {cfg.embeddings.model} + BM25", flush=True)
    embedder = ClinicalEmbedder()
    sparse_encoder = Bm25SparseEncoder()

    print(f"[2/4] Opening embedded Qdrant: {cfg.vector_store.local_path}", flush=True)
    client = get_qdrant_client()
    indexer = ICD10Indexer(client, embedder, sparse_encoder, on_progress=_print_progress)
    print(f"      -> collection '{indexer.collection_name}' ready", flush=True)

    print(f"[3/4] Indexing ICD-10 codes: {cfg.paths.icd10_processed_path}", flush=True)
    # skip_existing resumes an interrupted build; delete data/qdrant_data/
    # for a full rebuild (e.g. after changing the embedding model).
    code_count = indexer.index_from_jsonl(
        cfg.paths.icd10_processed_path, skip_existing=True
    )
    print(f"      -> {code_count:,} code documents upserted", flush=True)

    print(f"[4/4] Indexing guidelines corpus: {cfg.paths.guidelines_path}")
    chunks = load_guideline_chunks(cfg.paths.guidelines_path)
    guideline_count = indexer.index_guidelines(chunks)
    print(f"      -> {guideline_count} guideline sections upserted")

    total = client.count(indexer.collection_name).count
    client.close()  # release the embedded-storage lock cleanly
    elapsed = time.perf_counter() - started
    print(f"Done: {total:,} points in '{indexer.collection_name}' ({elapsed/60:.1f} min)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        print(
            "Hint: run `uv run python scripts/run_etl.py` first if the "
            "processed JSONL is missing.",
            file=sys.stderr,
        )
        sys.exit(1)
