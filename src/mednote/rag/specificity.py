"""Hierarchical specificity check (Step 7.5).

If a parent/unspecified code survives re-ranking (e.g. H65.9 "Unspecified
otitis media"), the physician should be offered its more specific children
(H65.91 right ear / H65.92 left ear / H65.93 bilateral) for final selection.
The family links collected by the ETL — and deliberately kept OUT of the
embedding text — pay off here: children are fetched from the index payloads
by exact code, never invented.
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from mednote.agent.schemas import SuggestedCode
from mednote.rag.indexer import DOC_TYPE_ICD10

_SOURCE = "ICD-10-CM 2026"


class SpecificityChecker:
    """Expands unspecified parent codes with their indexed children."""

    def __init__(self, client: QdrantClient, collection_name: str | None = None):
        from mednote.config import get_config

        self.client = client
        self.collection_name = collection_name or get_config().vector_store.collection_name

    def check_and_expand(self, candidates: list[dict]) -> list[SuggestedCode]:
        """Convert reranked candidates into SuggestedCode entries, attaching
        ``specificity_options`` wherever the code has indexed children."""
        return [self._to_suggested(candidate) for candidate in candidates]

    def _to_suggested(self, candidate: dict) -> SuggestedCode:
        children = candidate.get("children_codes") or []
        return {
            "code": candidate["code"],
            "description": candidate.get("description", ""),
            "hierarchy_path": candidate.get("hierarchy_path", ""),
            "source": _SOURCE,
            "confidence": candidate.get("confidence", 0.0),
            "parent_code": candidate.get("parent_code"),
            "specificity_options": self._fetch_children(children),
            "pending_confirmation": True,
        }

    def _fetch_children(self, children_codes: list[str]) -> list[SuggestedCode]:
        """Look up the children's payloads by exact code (no similarity)."""
        if not children_codes:
            return []

        points, _ = self.client.scroll(
            self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_type", match=models.MatchValue(value=DOC_TYPE_ICD10)
                    ),
                    models.FieldCondition(
                        key="code", match=models.MatchAny(any=children_codes)
                    ),
                ]
            ),
            limit=len(children_codes),
            with_payload=True,
        )
        options: list[SuggestedCode] = [
            {
                "code": p.payload["code"],
                "description": p.payload.get("description", ""),
                "hierarchy_path": p.payload.get("hierarchy_path", ""),
                "source": _SOURCE,
                "parent_code": p.payload.get("parent_code"),
                "pending_confirmation": True,
            }
            for p in points
        ]
        options.sort(key=lambda o: o["code"])
        return options
