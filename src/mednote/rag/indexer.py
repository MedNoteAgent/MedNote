"""Qdrant collection setup + batch upsert of dense/sparse vectors (Step 6.2).

No Docker, no server process: ``QdrantClient(path=...)`` runs Qdrant embedded
in-process and persists straight to disk (config.yml -> vector_store.local_path).

Single-process constraint: embedded local mode locks the data directory to one
process at a time. Run build_index.py / validate_index.py to completion before
starting the UI or eval scripts.

One collection holds both document types, distinguished by the ``doc_type``
payload field so the retriever can filter or blend them:

    icd10_code   ~47,000 self-contained ICD-10-CM code documents
    guideline    clinical documentation guideline sections (Step 6.4)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from qdrant_client import QdrantClient, models

from mednote.rag.etl.parser import ICD10Code
from mednote.rag.guidelines import GuidelineChunk

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DOC_TYPE_ICD10 = "icd10_code"
DOC_TYPE_GUIDELINE = "guideline"

# Payload fields the retriever hard-filters on (Task 7) get keyword indexes.
_KEYWORD_INDEX_FIELDS = ("doc_type", "code", "chapter_code", "target_sex")

_UPSERT_BATCH_SIZE = 512


class DenseEmbedder(Protocol):
    """Anything that encodes text into fixed-size dense vectors."""

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SparseEncoder(Protocol):
    """Anything that encodes text into Qdrant sparse vectors."""

    def encode(self, texts: list[str]) -> list[models.SparseVector]: ...


def get_qdrant_client(path: str | None = None) -> QdrantClient:
    """Embedded, local-disk Qdrant — no server, nothing to start or stop."""
    from mednote.config import get_config

    return QdrantClient(path=path or get_config().vector_store.local_path)


def ensure_collection(client: QdrantClient, name: str, dense_size: int) -> None:
    """Create the hybrid collection and payload indexes if they don't exist.

    - Dense vectors: cosine similarity (768-dim for SapBERT)
    - Sparse vectors: ``Modifier.IDF`` so Qdrant supplies the IDF half of
      BM25 to Bm25SparseEncoder's term-frequency weights
    - Keyword payload indexes for metadata hard-filtering (a no-op in
      embedded local mode, where filters scan payloads directly — kept so
      the schema is ready if the store ever moves to server mode)
    """
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_size, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
    for field in _KEYWORD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def _point_id(namespace: str, key: str) -> str:
    """Deterministic UUID so re-indexing overwrites instead of duplicating."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mednote/{namespace}/{key}"))


def _icd10_payload(code: ICD10Code) -> dict:
    """Flatten a code into the point payload the retriever reads (Task 7)."""
    return {
        "doc_type": DOC_TYPE_ICD10,
        "code": code.code,
        "description": code.description,
        "hierarchy_path": code.hierarchy_path,
        "chapter": code.chapter,
        "chapter_code": code.chapter_code,
        "parent_code": code.parent_code,
        "children_codes": code.children_codes,
        # [] means "no restriction"; normalize to ["all"] so the retriever's
        # MatchAny(["all", patient_sex]) filter never drops unrestricted codes.
        "target_sex": code.target_sex or ["all"],
        "max_age_days": code.max_age_days,
        "text": code.to_embedding_text(),
    }


def _read_jsonl(jsonl_path: Path) -> list[ICD10Code]:
    """Load the ETL hand-off artifact back into typed ICD10Code objects.

    Reconstructing the dataclass (rather than passing dicts through) fails
    fast on any schema drift between the ETL export and this indexer.
    """
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Processed ICD-10 JSONL not found: {jsonl_path}")

    codes = [
        ICD10Code(**json.loads(line))
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not codes:
        raise ValueError(f"Processed ICD-10 JSONL is empty: {jsonl_path}")
    return codes


class ICD10Indexer:
    """Embeds documents (dense + sparse) and batch-upserts them into Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: DenseEmbedder,
        sparse_encoder: SparseEncoder,
        collection_name: str | None = None,
        batch_size: int = _UPSERT_BATCH_SIZE,
        on_progress: Callable[[int, int], None] | None = None,
    ):
        from mednote.config import get_config

        self.client = client
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.collection_name = (
            collection_name or get_config().vector_store.collection_name
        )
        self.batch_size = batch_size
        self.on_progress = on_progress  # called with (documents done, total)
        ensure_collection(client, self.collection_name, embedder.dimension)

    def index_from_jsonl(
        self, jsonl_path: str | Path, skip_existing: bool = False
    ) -> int:
        """Embed + upsert every code from the ETL JSONL; returns the count.

        ``skip_existing=True`` resumes an interrupted build: codes whose
        deterministic point ID is already in the collection are not
        re-embedded. Only use it when the embedding model and payload schema
        are unchanged — a fresh model needs a full rebuild.
        """
        codes = _read_jsonl(Path(jsonl_path))
        if skip_existing:
            existing = self._existing_point_ids()
            codes = [c for c in codes if _point_id("icd10", c.code) not in existing]
            if not codes:
                return 0
        self._upsert(
            ids=[_point_id("icd10", c.code) for c in codes],
            texts=[c.to_embedding_text() for c in codes],
            payloads=[_icd10_payload(c) for c in codes],
        )
        return len(codes)

    def index_guidelines(self, chunks: list[GuidelineChunk]) -> int:
        """Embed + upsert guideline sections into the same collection."""
        if not chunks:
            raise ValueError("Refusing to index an empty guideline chunk list")
        self._upsert(
            ids=[_point_id("guideline", f"{c.source}#{c.heading}") for c in chunks],
            texts=[c.to_embedding_text() for c in chunks],
            payloads=[
                {
                    "doc_type": DOC_TYPE_GUIDELINE,
                    "heading": c.heading,
                    "source": c.source,
                    "target_sex": ["all"],  # guidelines apply to every patient
                    "text": c.to_embedding_text(),
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def _existing_point_ids(self) -> set[str]:
        """Scroll every point ID already in the collection (IDs only)."""
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection_name,
                limit=4096,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(str(p.id) for p in points)
            if offset is None:
                return ids

    def _upsert(self, ids: list[str], texts: list[str], payloads: list[dict]) -> None:
        """Encode and upsert one batch at a time.

        Streaming matters here: a full build is ~45 minutes of CPU embedding,
        and per-batch upserts mean an interrupted run keeps every batch it
        finished (resumable via ``skip_existing``) instead of losing all work.
        """
        total = len(ids)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_texts = texts[start:end]
            dense_vectors = self.embedder.embed(batch_texts)
            sparse_vectors = self.sparse_encoder.encode(batch_texts)

            points = [
                models.PointStruct(
                    id=point_id,
                    vector={
                        DENSE_VECTOR_NAME: dense.tolist()
                        if hasattr(dense, "tolist")
                        else list(dense),
                        SPARSE_VECTOR_NAME: sparse,
                    },
                    payload=payload,
                )
                for point_id, dense, sparse, payload in zip(
                    ids[start:end], dense_vectors, sparse_vectors, payloads[start:end],
                    strict=True,
                )
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            if self.on_progress is not None:
                self.on_progress(end, total)
