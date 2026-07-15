"""Hybrid retriever: dense + sparse fusion with demographic hard filters (Step 7.3).

Two searches run against the same collection and their votes are blended
(weights from config.yml -> vector_store, never hardcoded):

    dense  (weight 0.7)  SapBERT semantic similarity — catches paraphrases
    sparse (weight 0.3)  BM25 exact-term match — catches acronyms ("COPD")

Scores are min-max normalized within each result list before blending,
because cosine similarity (0-1) and BM25 (unbounded) live on different scales.

Demographic filters are walls, not weights: a demographically impossible code
(pregnancy code for a male patient, perinatal code for an adult) is excluded
BEFORE scoring, never merely down-ranked.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from mednote.rag.indexer import (
    DENSE_VECTOR_NAME,
    DOC_TYPE_ICD10,
    SPARSE_VECTOR_NAME,
)

_DAYS_PER_YEAR = 365


def _build_filter(
    patient_sex: str, patient_age: int | None, doc_type: str
) -> models.Filter:
    """Hard demographic filter; unknown demographics never exclude anything."""
    must: list[models.Condition] = [
        models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))
    ]
    if patient_sex in ("male", "female"):
        must.append(
            models.FieldCondition(
                key="target_sex", match=models.MatchAny(any=["all", patient_sex])
            )
        )
    if patient_age is not None:
        # Keep codes with no age cap (max_age_days null) OR a cap the patient
        # still satisfies (perinatal codes carry max_age_days=28).
        must.append(
            models.Filter(
                should=[
                    models.IsNullCondition(
                        is_null=models.PayloadField(key="max_age_days")
                    ),
                    models.FieldCondition(
                        key="max_age_days",
                        range=models.Range(gte=patient_age * _DAYS_PER_YEAR),
                    ),
                ]
            )
        )
    return models.Filter(must=must)


def _normalized(points: list) -> dict[str, tuple[float, dict]]:
    """Min-max normalize scores; returns {point_id: (norm_score, payload)}."""
    if not points:
        return {}
    scores = [p.score for p in points]
    low, high = min(scores), max(scores)
    span = high - low
    return {
        str(p.id): ((p.score - low) / span if span > 0 else 1.0, p.payload)
        for p in points
    }


class HybridRetriever:
    """Weighted dense+sparse retrieval over the Task 6 index."""

    def __init__(
        self,
        client: QdrantClient,
        embedder,
        sparse_encoder,
        collection_name: str | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        top_k: int | None = None,
    ):
        from mednote.config import get_config

        cfg = get_config().vector_store
        self.client = client
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.collection_name = collection_name or cfg.collection_name
        self.dense_weight = cfg.dense_weight if dense_weight is None else dense_weight
        self.sparse_weight = (
            cfg.sparse_weight if sparse_weight is None else sparse_weight
        )
        self.top_k = top_k or cfg.top_k_retrieve

    def retrieve(
        self,
        query: str,
        patient_sex: str = "unknown",
        patient_age: int | None = None,
        top_k: int | None = None,
        doc_type: str = DOC_TYPE_ICD10,
    ) -> list[dict]:
        """Return up to ``top_k`` candidate payloads with a fused ``score``.

        Raises:
            ValueError: if the query is blank.
        """
        if not query or not query.strip():
            raise ValueError("Retrieval query is empty")

        limit = top_k or self.top_k
        query_filter = _build_filter(patient_sex, patient_age, doc_type)

        dense = _normalized(self._search(
            self.embedder.embed_query(query), DENSE_VECTOR_NAME, query_filter, limit
        ))
        sparse = _normalized(self._search(
            self.sparse_encoder.encode_query(query), SPARSE_VECTOR_NAME, query_filter, limit
        ))

        fused: list[dict] = []
        for point_id in dense.keys() | sparse.keys():
            dense_score, payload = dense.get(point_id, (0.0, None))
            sparse_score, sparse_payload = sparse.get(point_id, (0.0, None))
            fused.append(
                {
                    **(payload or sparse_payload),
                    "score": self.dense_weight * dense_score
                    + self.sparse_weight * sparse_score,
                }
            )
        fused.sort(key=lambda c: c["score"], reverse=True)
        return fused[:limit]

    def _search(self, query, using: str, query_filter: models.Filter, limit: int) -> list:
        return self.client.query_points(
            self.collection_name,
            query=query,
            using=using,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        ).points
