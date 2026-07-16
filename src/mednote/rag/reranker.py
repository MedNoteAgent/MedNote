"""Cross-encoder re-ranking of retrieval candidates (Step 7.4).

Vector search is good at recall but bad at precision: the top-15 candidates
reliably contain the right code, rarely at rank 1. The cross-encoder reads
the query and each candidate's text *together* (full attention across the
pair, not two independent vectors) and produces a much sharper relevance
score. Raw logits are squashed through a sigmoid so the confidence is a
0-1 value comparable against config.yml's ``confidence_threshold``.
"""

from __future__ import annotations

import math


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class ClinicalReranker:
    """Scores (query, candidate text) pairs with a cross-encoder."""

    def __init__(self, model=None, model_name: str | None = None):
        if model is None:
            from sentence_transformers import CrossEncoder

            from mednote.config import get_config

            model = CrossEncoder(model_name or get_config().reranker.model)
        self._model = model

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        """Return the ``top_n`` candidates as NEW dicts with ``confidence``.

        Each candidate must carry a ``text`` field (the Task 6 payload always
        does). Input dicts are not mutated.
        """
        if not candidates:
            return []

        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self._model.predict(pairs)

        scored = [
            {**candidate, "confidence": _sigmoid(float(score))}
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda c: c["confidence"], reverse=True)
        return scored[:top_n]
