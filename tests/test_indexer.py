"""Tests for Task 6: embedding & vector DB indexing.

Unit tests run against an in-memory Qdrant instance with fake dense/sparse
encoders, so they never download SapBERT or the BM25 tokenizer. The fakes are
deterministic: identical text always produces the identical vector, which lets
the search-roundtrip test assert exact-match retrieval through the real Qdrant
query path (named vectors, filters, payload).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from mednote.rag.etl.parser import ICD10Code
from mednote.rag.guidelines import GuidelineChunk, load_guideline_chunks
from mednote.rag.indexer import (
    DENSE_VECTOR_NAME,
    DOC_TYPE_GUIDELINE,
    DOC_TYPE_ICD10,
    SPARSE_VECTOR_NAME,
    ICD10Indexer,
    ensure_collection,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_GUIDELINES = REPO_ROOT / "data" / "corpus" / "clinical_guidelines.md"


class FakeEmbedder:
    """Deterministic stand-in for ClinicalEmbedder (no model download)."""

    dimension = 8

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self._vector(t) for t in texts], dtype=np.float32)

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: FakeEmbedder.dimension]]


class FakeSparseEncoder:
    """Deterministic stand-in for Bm25SparseEncoder: one index per token."""

    def encode(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._sparse(t) for t in texts]

    def encode_query(self, query: str) -> models.SparseVector:
        return self._sparse(query)

    @staticmethod
    def _sparse(text: str) -> models.SparseVector:
        counts: dict[int, float] = {}
        for token in text.lower().split():
            idx = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return models.SparseVector(indices=list(counts), values=list(counts.values()))


SAMPLE_CODES = [
    ICD10Code(
        code="I21.9",
        description="Acute myocardial infarction, unspecified",
        hierarchy_path="Circulatory -> Ischemic heart diseases",
        chapter="Diseases of the circulatory system",
        chapter_code="9",
        index_synonyms=["Infarct, infarction, myocardium"],
    ),
    ICD10Code(
        code="O80",
        description="Encounter for full-term uncomplicated delivery",
        hierarchy_path="Pregnancy -> Delivery",
        chapter="Pregnancy, childbirth and the puerperium",
        chapter_code="15",
        target_sex=["female"],
    ),
    ICD10Code(
        code="J44.1",
        description="Chronic obstructive pulmonary disease with acute exacerbation",
        hierarchy_path="Respiratory -> Chronic lower respiratory diseases",
        chapter="Diseases of the respiratory system",
        chapter_code="10",
        inclusion_terms=["COPD with acute exacerbation"],
    ),
]


@pytest.fixture()
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture()
def jsonl_path(tmp_path: Path) -> Path:
    path = tmp_path / "codes.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for code in SAMPLE_CODES:
            handle.write(json.dumps(asdict(code)) + "\n")
    return path


@pytest.fixture()
def indexer(client: QdrantClient) -> ICD10Indexer:
    return ICD10Indexer(
        client=client,
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        collection_name="test_codes",
    )


# ------------------------------------------------------------- collection ---


def test_ensure_collection_provisions_dense_sparse_and_indexes(
    client: QdrantClient,
) -> None:
    ensure_collection(client, "test_codes", dense_size=8)

    info = client.get_collection("test_codes")
    dense = info.config.params.vectors[DENSE_VECTOR_NAME]
    assert dense.size == 8
    assert dense.distance == models.Distance.COSINE

    sparse = info.config.params.sparse_vectors[SPARSE_VECTOR_NAME]
    assert sparse.modifier == models.Modifier.IDF


def test_ensure_collection_is_idempotent(client: QdrantClient) -> None:
    ensure_collection(client, "test_codes", dense_size=8)
    ensure_collection(client, "test_codes", dense_size=8)  # must not raise
    assert client.collection_exists("test_codes")


# ----------------------------------------------------------- ICD indexing ---


def test_index_from_jsonl_upserts_every_record(
    indexer: ICD10Indexer, client: QdrantClient, jsonl_path: Path
) -> None:
    count = indexer.index_from_jsonl(jsonl_path)
    assert count == len(SAMPLE_CODES)
    assert client.count("test_codes").count == len(SAMPLE_CODES)


def test_index_from_jsonl_payload_carries_metadata(
    indexer: ICD10Indexer, client: QdrantClient, jsonl_path: Path
) -> None:
    indexer.index_from_jsonl(jsonl_path)

    points, _ = client.scroll("test_codes", limit=10, with_payload=True)
    by_code = {p.payload["code"]: p.payload for p in points}

    i219 = by_code["I21.9"]
    assert i219["doc_type"] == DOC_TYPE_ICD10
    assert i219["description"] == "Acute myocardial infarction, unspecified"
    assert i219["chapter_code"] == "9"
    # Unrestricted codes are normalized to ["all"] so the retriever's
    # MatchAny(["all", patient_sex]) filter never drops them.
    assert i219["target_sex"] == ["all"]
    assert "Infarct" in i219["text"]

    assert by_code["O80"]["target_sex"] == ["female"]


def test_index_from_jsonl_is_idempotent(
    indexer: ICD10Indexer, client: QdrantClient, jsonl_path: Path
) -> None:
    indexer.index_from_jsonl(jsonl_path)
    indexer.index_from_jsonl(jsonl_path)  # same deterministic IDs -> overwrite
    assert client.count("test_codes").count == len(SAMPLE_CODES)


def test_index_from_jsonl_streams_in_batches_with_progress(
    client: QdrantClient, jsonl_path: Path
) -> None:
    """Each batch is embedded AND upserted before the next starts, so an
    interrupted build keeps everything already reported by the callback."""
    progress: list[tuple[int, int]] = []
    indexer = ICD10Indexer(
        client=client,
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        collection_name="test_codes",
        batch_size=2,
        on_progress=lambda done, total: progress.append((done, total)),
    )
    indexer.index_from_jsonl(jsonl_path)
    assert progress == [(2, 3), (3, 3)]


def test_index_from_jsonl_skip_existing_resumes_interrupted_build(
    indexer: ICD10Indexer, client: QdrantClient, jsonl_path: Path, tmp_path: Path
) -> None:
    indexer.index_from_jsonl(jsonl_path)
    # Re-run over the same JSONL: everything is already indexed.
    assert indexer.index_from_jsonl(jsonl_path, skip_existing=True) == 0

    # A JSONL with one extra code indexes only the new record.
    extra = ICD10Code(
        code="G44.2",
        description="Tension-type headache",
        hierarchy_path="Nervous -> Episodic",
        chapter="Diseases of the nervous system",
        chapter_code="6",
    )
    grown = tmp_path / "grown.jsonl"
    with grown.open("w", encoding="utf-8") as handle:
        for code in [*SAMPLE_CODES, extra]:
            handle.write(json.dumps(asdict(code)) + "\n")

    assert indexer.index_from_jsonl(grown, skip_existing=True) == 1
    assert client.count("test_codes").count == len(SAMPLE_CODES) + 1


def test_index_from_jsonl_rejects_missing_file(indexer: ICD10Indexer, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        indexer.index_from_jsonl(tmp_path / "nope.jsonl")


def test_index_from_jsonl_rejects_empty_file(indexer: ICD10Indexer, tmp_path: Path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        indexer.index_from_jsonl(empty)


def test_indexed_codes_are_retrievable_by_dense_vector(
    indexer: ICD10Indexer, client: QdrantClient, jsonl_path: Path
) -> None:
    """Roundtrip through the real Qdrant query path with named vectors."""
    indexer.index_from_jsonl(jsonl_path)

    target_text = next(c for c in SAMPLE_CODES if c.code == "J44.1").to_embedding_text()
    hits = client.query_points(
        "test_codes",
        query=FakeEmbedder().embed_query(target_text),
        using=DENSE_VECTOR_NAME,
        limit=1,
    ).points
    assert hits[0].payload["code"] == "J44.1"


def test_indexed_codes_are_retrievable_by_sparse_vector(
    indexer: ICD10Indexer, client: QdrantClient, jsonl_path: Path
) -> None:
    indexer.index_from_jsonl(jsonl_path)

    hits = client.query_points(
        "test_codes",
        query=FakeSparseEncoder().encode_query("COPD"),
        using=SPARSE_VECTOR_NAME,
        limit=1,
    ).points
    assert hits[0].payload["code"] == "J44.1"


# -------------------------------------------------------------- guidelines ---

GUIDELINES_FIXTURE = """# Clinical Documentation Guidelines

Intro preamble that belongs to no section.

## SOAP Note Structure

Subjective, Objective, Assessment, Plan.

## Red-Flag Symptom Combinations

Chest pain with radiation to the left arm requires urgent escalation.
"""


def test_load_guideline_chunks_splits_by_section_heading(tmp_path: Path) -> None:
    path = tmp_path / "guidelines.md"
    path.write_text(GUIDELINES_FIXTURE, encoding="utf-8")

    chunks = load_guideline_chunks(path)
    assert [c.heading for c in chunks] == [
        "SOAP Note Structure",
        "Red-Flag Symptom Combinations",
    ]
    assert "urgent escalation" in chunks[1].text
    assert chunks[1].source == "guidelines.md"


def test_load_guideline_chunks_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_guideline_chunks(tmp_path / "nope.md")


def test_guideline_chunk_embedding_text_includes_heading() -> None:
    chunk = GuidelineChunk(heading="Red Flags", text="Escalate chest pain.", source="g.md")
    assert chunk.to_embedding_text() == "Red Flags\nEscalate chest pain."


def test_index_guidelines_tags_doc_type(
    indexer: ICD10Indexer, client: QdrantClient
) -> None:
    chunks = [
        GuidelineChunk(heading="SOAP Note Structure", text="S O A P.", source="g.md"),
        GuidelineChunk(heading="Red Flags", text="Escalate chest pain.", source="g.md"),
    ]
    count = indexer.index_guidelines(chunks)
    assert count == 2

    points, _ = client.scroll("test_codes", limit=10, with_payload=True)
    payloads = [p.payload for p in points]
    assert all(p["doc_type"] == DOC_TYPE_GUIDELINE for p in payloads)
    assert {p["heading"] for p in payloads} == {"SOAP Note Structure", "Red Flags"}


def test_index_guidelines_rejects_empty_list(indexer: ICD10Indexer) -> None:
    with pytest.raises(ValueError, match="empty"):
        indexer.index_guidelines([])


def test_icd_and_guidelines_share_one_collection_filterable_by_doc_type(
    indexer: ICD10Indexer, client: QdrantClient, jsonl_path: Path
) -> None:
    indexer.index_from_jsonl(jsonl_path)
    indexer.index_guidelines(
        [GuidelineChunk(heading="Red Flags", text="Escalate.", source="g.md")]
    )

    total = client.count("test_codes").count
    assert total == len(SAMPLE_CODES) + 1

    guideline_only = client.count(
        "test_codes",
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_type", match=models.MatchValue(value=DOC_TYPE_GUIDELINE)
                )
            ]
        ),
    ).count
    assert guideline_only == 1


# ------------------------------------------------------------ integration ---


@pytest.mark.skipif(
    not REAL_GUIDELINES.exists(),
    reason="clinical_guidelines.md not present in data/corpus/",
)
def test_real_guidelines_corpus_chunks_cleanly() -> None:
    chunks = load_guideline_chunks(REAL_GUIDELINES)
    assert len(chunks) == 6
    headings = [c.heading for c in chunks]
    assert "SOAP Note Structure" in headings
    assert any("Red-Flag" in h for h in headings)
    assert all(c.text.strip() for c in chunks)
