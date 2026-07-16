"""Deterministic test doubles shared across RAG test modules.

All fakes are hash-based and deterministic: identical text always produces the
identical vector/score, so tests can assert exact-match retrieval through real
Qdrant query paths without downloading any model.
"""

from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace

import numpy as np
from qdrant_client import models


class FakeEmbedder:
    """Stand-in for ClinicalEmbedder: same text -> same vector."""

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
    """Stand-in for Bm25SparseEncoder: one sparse index per distinct token."""

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


class FakeChatModel:
    """Stand-in for a LangChain chat model: returns canned .content replies."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.prompts: list[str] = []  # records what it was asked, for asserts

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("FakeChatModel ran out of canned responses")
        return SimpleNamespace(content=self._responses.pop(0))


class FakeCrossEncoder:
    """Stand-in for a CrossEncoder: logit = scaled token overlap (Jaccard).

    Disjoint pair -> logit -2 (sigmoid ~0.12, below the 0.7 threshold);
    heavily overlapping pair -> positive logit (sigmoid above threshold).
    """

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._logit(query, text) for query, text in pairs]

    @staticmethod
    def _logit(query: str, text: str) -> float:
        q_tokens = set(query.lower().split())
        t_tokens = set(text.lower().split())
        if not q_tokens or not t_tokens:
            return -2.0
        jaccard = len(q_tokens & t_tokens) / len(q_tokens | t_tokens)
        return -2.0 + 16.0 * jaccard


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
