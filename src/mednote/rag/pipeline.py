"""Full RAG pipeline orchestration (Task 7).

    RAGPipeline.run(assessment, sex, age) -> list[SuggestedCode]
        ├── entity_extractor.extract(assessment)         -> entities[]
        ├── per entity: cache check -> hybrid retrieval  -> candidates[]
        │               reranker.rerank(entity, ...)     -> confidence per code
        ├── merge entities' results, best confidence per code, top_k overall
        ├── zero-hit check: max confidence < threshold   -> [] (degradation)
        └── specificity_checker.check_and_expand(...)    -> list[SuggestedCode]

Demographics arrive as parameters: the LangGraph context-extraction node
(Task 8) reads them from the mock EHR — this pipeline never touches the EHR
itself. The cache key includes the demographics because the hard filters
change what retrieval returns for the same entity.

Zero-Hit Protocol: an empty return means no candidate cleared
``confidence_threshold``; the calling node surfaces ZERO_HIT_MESSAGE instead
of guessing a code.
"""

from __future__ import annotations

import logging

from mednote.agent.schemas import SuggestedCode
from mednote.rag.cache import RAGCache
from mednote.rag.entity_extractor import EntityExtractor
from mednote.rag.reranker import ClinicalReranker
from mednote.rag.retriever import HybridRetriever
from mednote.rag.specificity import SpecificityChecker

logger = logging.getLogger(__name__)

ZERO_HIT_MESSAGE = (
    "Insufficient data to suggest an accurate ICD-10 code. "
    "Please manually assign in EHR."
)


class RAGPipeline:
    """Wires extractor -> cache -> retriever -> reranker -> specificity."""

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        retriever: HybridRetriever,
        reranker: ClinicalReranker,
        specificity_checker: SpecificityChecker,
        cache: RAGCache | None = None,
    ):
        from mednote.config import get_config

        cfg = get_config().vector_store
        self.entity_extractor = entity_extractor
        self.retriever = retriever
        self.reranker = reranker
        self.specificity_checker = specificity_checker
        self.cache = cache or RAGCache()
        self.top_k_rerank = cfg.top_k_rerank
        self.confidence_threshold = cfg.confidence_threshold

    def run(
        self,
        assessment_text: str,
        patient_sex: str = "unknown",
        patient_age: int | None = None,
        entities: list[str] | None = None,
    ) -> list[SuggestedCode]:
        """Suggest ICD-10 codes for a SOAP assessment; [] means zero-hit.

        ``entities``: pass pre-extracted entities (e.g. from the graph's
        entity_extraction node) to skip internal extraction — avoids a second
        LLM call when extraction already happened upstream.

        Raises:
            ValueError: if the assessment text is blank.
        """
        if not assessment_text or not assessment_text.strip():
            raise ValueError("Assessment text is empty")

        if entities is None:
            entities = self._extract_entities(assessment_text)
        top = self._retrieve_and_rerank(entities, patient_sex, patient_age)
        if not top or top[0]["confidence"] < self.confidence_threshold:
            return []  # zero-hit: caller surfaces ZERO_HIT_MESSAGE

        return self.specificity_checker.check_and_expand(top)

    def _extract_entities(self, assessment_text: str) -> list[str]:
        """Extract normalized entities; degrade to the raw assessment text."""
        try:
            entities = self.entity_extractor.extract(assessment_text)
            if entities:
                return entities
            logger.warning("Entity extraction returned no entities; using raw assessment")
        except ValueError as exc:
            logger.warning("Entity extraction failed (%s); using raw assessment", exc)
        return [assessment_text.strip()]

    def _retrieve_and_rerank(
        self, entities: list[str], patient_sex: str, patient_age: int | None
    ) -> list[dict]:
        """Retrieve (cache-first) and rerank PER ENTITY, then merge by code.

        Each candidate is scored against the normalized entity that retrieved
        it — never the raw assessment, and never a join of all entities.
        Measured on the real index: the cross-encoder scores formal code text
        near zero against colloquial phrasing ("heart attack" -> I21.9 at
        0.004, a guaranteed zero-hit), and joining multiple entities dilutes
        every candidate's score ("Shortness of breath; Chest tightness" vs
        R06.02 -> 0.425 while the single entity clears the threshold).
        Per-entity scoring keeps the vocabulary gap closed on both sides.
        """
        by_code: dict[str, dict] = {}
        entity_best: list[str] = []  # each entity's top code — guaranteed a slot
        for entity in entities:
            cache_key = f"{entity}|sex={patient_sex}|age={patient_age}"
            candidates = self.cache.get(cache_key)
            if candidates is None:
                candidates = self.retriever.retrieve(
                    entity, patient_sex=patient_sex, patient_age=patient_age
                )
                self.cache.set(cache_key, candidates)
            reranked = self.reranker.rerank(entity, candidates, self.top_k_rerank)
            # Guarantee a slot only for credible codes: a sub-threshold best
            # (e.g. "Prehypertension" -> O10.03 at 0.39, measured live) is
            # noise the physician shouldn't see, not a condition to protect.
            if reranked and reranked[0]["confidence"] >= self.confidence_threshold:
                entity_best.append(reranked[0]["code"])
            for scored in reranked:
                code = scored["code"]
                # Keep the best confidence when entities overlap on a code.
                if (
                    code not in by_code
                    or scored["confidence"] > by_code[code]["confidence"]
                ):
                    by_code[code] = scored

        # Global cap with per-entity fairness: one entity's high-scoring
        # candidates must not crowd another condition out entirely (measured
        # live: three extracted conditions -> a top-3 of only insomnia codes,
        # dropping the chief complaint). Each entity's best code is reserved
        # a slot; remaining slots fill by confidence.
        merged = sorted(by_code.values(), key=lambda c: c["confidence"], reverse=True)
        guaranteed = set(entity_best)
        limit = max(self.top_k_rerank, len(guaranteed))
        kept = [c for c in merged if c["code"] in guaranteed]
        kept.extend(c for c in merged if c["code"] not in guaranteed)
        return sorted(kept[:limit], key=lambda c: c["confidence"], reverse=True)
