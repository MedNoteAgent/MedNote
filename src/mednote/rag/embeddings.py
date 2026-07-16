"""Text -> vector encoders for indexing and retrieval (Step 6.1).

Two encoders feed the hybrid index:

    ClinicalEmbedder    dense 768-dim SapBERT vectors (semantic similarity)
    Bm25SparseEncoder   BM25 term-frequency sparse vectors (exact acronyms)

Why SapBERT over generic models (docs/implementation_plan.md Task 6): it is
trained on UMLS concept pairs, so "heart attack" and "acute myocardial
infarction" land close together — generic sentence encoders miss medical
synonym relationships entirely.
"""

from __future__ import annotations

import numpy as np
from qdrant_client import models


class ClinicalEmbedder:
    """SapBERT wrapper — SOTA for mapping medical synonyms to ontologies."""

    def __init__(self, model_name: str | None = None, batch_size: int | None = None):
        from sentence_transformers import SentenceTransformer

        from mednote.config import get_config

        cfg = get_config().embeddings
        self.model = SentenceTransformer(model_name or cfg.model)
        self.batch_size = batch_size or cfg.batch_size

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode documents; SapBERT handles internal batching.

        No progress bar: the indexer streams small batches and reports its
        own progress, so a per-call bar would just flicker.
        """
        return self.model.encode(
            texts, show_progress_bar=False, batch_size=self.batch_size
        )

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query], show_progress_bar=False)[0].tolist()


class Bm25SparseEncoder:
    """BM25 sparse vectors via fastembed's Qdrant/bm25 model.

    Emits term-frequency weights; the collection's ``Modifier.IDF`` supplies
    the IDF half of BM25 at query time (see indexer.ensure_collection). This
    is what catches exact acronym matches like "COPD" that dense embeddings
    can blur.
    """

    def __init__(self, model_name: str = "Qdrant/bm25"):
        from fastembed import SparseTextEmbedding

        self.model = SparseTextEmbedding(model_name=model_name)

    def encode(self, texts: list[str]) -> list[models.SparseVector]:
        """Document-side encoding (applies BM25 term-frequency saturation)."""
        return [
            models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self.model.embed(texts)
        ]

    def encode_query(self, query: str) -> models.SparseVector:
        """Query-side encoding (raw term presence, no TF saturation)."""
        embedding = next(iter(self.model.query_embed(query)))
        return models.SparseVector(
            indices=embedding.indices.tolist(), values=embedding.values.tolist()
        )
