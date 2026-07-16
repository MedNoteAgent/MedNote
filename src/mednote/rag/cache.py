"""LRU cache for repeated ICD-10 RAG lookups (dict-based, no Redis).

A physician session repeats queries constantly ("tension headache" across
several patients in one clinic day); a cache hit skips embedding + Qdrant
entirely (<5 ms vs ~300-600 ms). Keys are normalized (lowercased, stripped)
so trivial formatting differences still hit.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import md5


class RAGCache:
    """LRU cache mapping a query string to its retrieval candidates."""

    def __init__(self, max_size: int | None = None):
        from mednote.config import get_config

        self._cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._max_size = max_size or get_config().cache.rag_max_size
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(query: str) -> str:
        return md5(query.lower().strip().encode()).hexdigest()

    def get(self, query: str) -> list[dict] | None:
        key = self._key(query)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def set(self, query: str, results: list[dict]) -> None:
        key = self._key(query)
        self._cache[key] = results
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "size": len(self._cache),
        }
