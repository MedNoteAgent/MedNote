"""Sanity-check the built Qdrant index (Step 6.3 / 6.4).

Checks (docs/implementation_plan.md Task 6 Definition of Done):
    1. Point count matches the ETL JSONL plus the guideline sections
    2. Dense:  "heart attack" -> an I21* acute-MI code in the top 3
    3. Dense:  "acute myocardial infarction" (the normalized entity the
       retriever actually receives, Step 7.2) -> I21.9 in the top 3
    4. Sparse: "COPD" -> a J44.* code in the top 5 (BM25 exact acronym match)
    5. Guideline: chest-pain escalation query surfaces the red-flag section

Usage (from the repo root, after build_index.py has exited):
    uv run python scripts/validate_index.py
"""

from __future__ import annotations

import sys

from qdrant_client import QdrantClient, models

from mednote.config import get_config
from mednote.rag.embeddings import Bm25SparseEncoder, ClinicalEmbedder
from mednote.rag.indexer import (
    DENSE_VECTOR_NAME,
    DOC_TYPE_GUIDELINE,
    DOC_TYPE_ICD10,
    SPARSE_VECTOR_NAME,
    get_qdrant_client,
)


def _doc_type_filter(doc_type: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))]
    )


def _top_payloads(
    client: QdrantClient,
    collection: str,
    query,
    using: str,
    doc_type: str,
    limit: int,
) -> list[dict]:
    hits = client.query_points(
        collection,
        query=query,
        using=using,
        query_filter=_doc_type_filter(doc_type),
        limit=limit,
    ).points
    return [h.payload for h in hits]


def check_point_count(client: QdrantClient, collection: str) -> tuple[bool, str]:
    cfg = get_config()
    expected_codes = sum(
        1 for line in open(cfg.paths.icd10_processed_path, encoding="utf-8") if line.strip()
    )
    total = client.count(collection).count
    ok = total >= expected_codes  # codes + at least one guideline section
    return ok, f"point count: {total:,} (expected >= {expected_codes:,} codes)"


def check_dense_heart_attack(
    client: QdrantClient, collection: str, embedder: ClinicalEmbedder
) -> tuple[bool, str]:
    """Colloquial query: an acute-MI code (I21*) must reach the top 3.

    The ICD-10 Index routes "heart attack" through a code-less *see*
    cross-reference, so no code document contains the literal phrase — at
    runtime the entity extractor (Step 7.2) normalizes it before retrieval.
    This check asserts SapBERT still lands the colloquialism on the right
    code family unaided.
    """
    payloads = _top_payloads(
        client,
        collection,
        query=embedder.embed_query("heart attack"),
        using=DENSE_VECTOR_NAME,
        doc_type=DOC_TYPE_ICD10,
        limit=3,
    )
    codes = [p["code"] for p in payloads]
    ok = any(c.startswith("I21") for c in codes)
    return ok, f'dense "heart attack" top-3: {codes}'


def check_dense_normalized_mi(
    client: QdrantClient, collection: str, embedder: ClinicalEmbedder
) -> tuple[bool, str]:
    """Normalized entity (what the retriever actually receives): I21.9 top-3."""
    payloads = _top_payloads(
        client,
        collection,
        query=embedder.embed_query("acute myocardial infarction"),
        using=DENSE_VECTOR_NAME,
        doc_type=DOC_TYPE_ICD10,
        limit=3,
    )
    codes = [p["code"] for p in payloads]
    return "I21.9" in codes, f'dense "acute myocardial infarction" top-3: {codes}'


def check_sparse_copd(
    client: QdrantClient, collection: str, sparse_encoder: Bm25SparseEncoder
) -> tuple[bool, str]:
    payloads = _top_payloads(
        client,
        collection,
        query=sparse_encoder.encode_query("COPD"),
        using=SPARSE_VECTOR_NAME,
        doc_type=DOC_TYPE_ICD10,
        limit=5,
    )
    codes = [p["code"] for p in payloads]
    return any(c.startswith("J44") for c in codes), f'sparse "COPD" top-5: {codes}'


def check_guideline_red_flag(
    client: QdrantClient, collection: str, embedder: ClinicalEmbedder
) -> tuple[bool, str]:
    payloads = _top_payloads(
        client,
        collection,
        query=embedder.embed_query("how do I escalate chest pain with arm radiation?"),
        using=DENSE_VECTOR_NAME,
        doc_type=DOC_TYPE_GUIDELINE,
        limit=1,
    )
    headings = [p["heading"] for p in payloads]
    ok = any("red-flag" in h.lower() for h in headings)
    return ok, f"guideline escalation query top-1: {headings}"


def main() -> int:
    cfg = get_config()
    collection = cfg.vector_store.collection_name
    client = get_qdrant_client()
    if not client.collection_exists(collection):
        client.close()
        print(f"Collection '{collection}' does not exist — run build_index.py first.")
        return 1

    embedder = ClinicalEmbedder()
    sparse_encoder = Bm25SparseEncoder()

    results = [
        check_point_count(client, collection),
        check_dense_heart_attack(client, collection, embedder),
        check_dense_normalized_mi(client, collection, embedder),
        check_sparse_copd(client, collection, sparse_encoder),
        check_guideline_red_flag(client, collection, embedder),
    ]
    for ok, message in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {message}")

    failed = sum(1 for ok, _ in results if not ok)
    client.close()  # release the embedded-storage lock cleanly
    print(f"{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
