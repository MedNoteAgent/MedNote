"""Tests for Task 7 orchestration: entity extraction, RAG cache, and the
full RAGPipeline (with fake LLM / encoders / cross-encoder throughout)."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from qdrant_client import QdrantClient

from tests.fakes import FakeChatModel, FakeCrossEncoder, FakeEmbedder, FakeSparseEncoder
from tests.test_rag_retrieval import BY_CODE, CODES

from mednote.rag.cache import RAGCache
from mednote.rag.entity_extractor import EntityExtractor
from mednote.rag.indexer import ICD10Indexer
from mednote.rag.pipeline import ZERO_HIT_MESSAGE, RAGPipeline
from mednote.rag.reranker import ClinicalReranker
from mednote.rag.retriever import HybridRetriever
from mednote.rag.specificity import SpecificityChecker

COLLECTION = "test_pipeline"


# ------------------------------------------------------------------ cache ---


def test_cache_returns_stored_results_and_counts_hits() -> None:
    cache = RAGCache(max_size=4)
    assert cache.get("tension headache") is None
    cache.set("tension headache", [{"code": "G44.2"}])
    assert cache.get("Tension Headache  ") == [{"code": "G44.2"}]  # normalized key
    assert cache.hits == 1
    assert cache.misses == 1
    assert cache.hit_rate == 0.5


def test_cache_evicts_least_recently_used() -> None:
    cache = RAGCache(max_size=2)
    cache.set("a", [{"code": "A"}])
    cache.set("b", [{"code": "B"}])
    cache.get("a")  # touch "a" so "b" is now least recently used
    cache.set("c", [{"code": "C"}])

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None


# ------------------------------------------------------- entity extraction ---


def test_extractor_parses_json_array_from_llm() -> None:
    llm = FakeChatModel(['["Acute bilateral otitis media", "Essential hypertension"]'])
    entities = EntityExtractor(llm=llm).extract("kid has an ear infection in both ears")
    assert entities == ["Acute bilateral otitis media", "Essential hypertension"]
    # The prompt must carry the assessment text, not the whole transcript.
    assert "ear infection in both ears" in llm.prompts[0]


def test_extractor_strips_markdown_code_fences() -> None:
    llm = FakeChatModel(['```json\n["Tension-type headache"]\n```'])
    assert EntityExtractor(llm=llm).extract("recurrent tension headache") == [
        "Tension-type headache"
    ]


def test_extractor_drops_blank_entities() -> None:
    llm = FakeChatModel(['["Hypertension", "", "  "]'])
    assert EntityExtractor(llm=llm).extract("bp high") == ["Hypertension"]


def test_extractor_handles_content_block_lists() -> None:
    """LangChain's Gemini wrapper returns content as a list of blocks, not str."""

    class BlockLLM:
        def invoke(self, prompt):
            from types import SimpleNamespace

            return SimpleNamespace(
                content=[{"type": "text", "text": '["Essential hypertension"]'}]
            )

    assert EntityExtractor(llm=BlockLLM()).extract("bp high") == ["Essential hypertension"]


def test_extractor_rejects_unparseable_output() -> None:
    llm = FakeChatModel(["The conditions are hypertension and diabetes."])
    with pytest.raises(ValueError):
        EntityExtractor(llm=llm).extract("assessment")


def test_extractor_rejects_non_string_array() -> None:
    llm = FakeChatModel(['{"entities": ["x"]}'])
    with pytest.raises(ValueError):
        EntityExtractor(llm=llm).extract("assessment")


def test_extractor_rejects_empty_assessment() -> None:
    with pytest.raises(ValueError):
        EntityExtractor(llm=FakeChatModel([])).extract("   ")


def test_extraction_prompt_pins_documented_findings_contract() -> None:
    """The prompt is a contract: extraction must not treat a clinician's
    screening question as a confirmed finding, and must not upgrade symptoms
    to inferred diagnoses (measured failure: chest-pain transcript ->
    I20.9/I24.81 angina codes plus dyspnea the patient never confirmed)."""
    from mednote.rag.entity_extractor import ENTITY_EXTRACTION_PROMPT

    lower = ENTITY_EXTRACTION_PROMPT.lower()
    # Rule: asked-about symptoms are not findings until confirmed.
    assert "asks about" in lower
    assert "confirm" in lower
    assert "denied" in lower
    # Rule: no diagnosis inference — symptom terms stay symptom-level.
    assert "do not infer" in lower
    assert '"chest pain", not "angina pectoris"' in lower
    # The negative example must demonstrate both rules end-to-end.
    assert "shortness of breath" in lower
    assert '["Chest pain", "Nausea", "Diaphoresis"]' in ENTITY_EXTRACTION_PROMPT
    # Normalization and output format survive the edit.
    assert "standard clinical terminology" in lower
    assert "json array of strings" in lower
    assert "{assessment_text}" in ENTITY_EXTRACTION_PROMPT


# --------------------------------------------------------------- pipeline ---


class SpyRetriever(HybridRetriever):
    """Counts retrieve() calls so cache behavior can be asserted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def retrieve(self, *args, **kwargs):
        self.calls += 1
        return super().retrieve(*args, **kwargs)


@pytest.fixture()
def indexed_client(tmp_path) -> QdrantClient:
    client = QdrantClient(":memory:")
    jsonl = tmp_path / "codes.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for code in CODES:
            handle.write(json.dumps(asdict(code)) + "\n")
    ICD10Indexer(
        client=client,
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        collection_name=COLLECTION,
    ).index_from_jsonl(jsonl)
    return client


def _make_pipeline(client: QdrantClient, llm_responses: list[str]) -> tuple[RAGPipeline, SpyRetriever]:
    retriever = SpyRetriever(
        client=client,
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        collection_name=COLLECTION,
    )
    pipeline = RAGPipeline(
        entity_extractor=EntityExtractor(llm=FakeChatModel(llm_responses)),
        retriever=retriever,
        reranker=ClinicalReranker(model=FakeCrossEncoder()),
        specificity_checker=SpecificityChecker(client=client, collection_name=COLLECTION),
        cache=RAGCache(max_size=8),
    )
    return pipeline, retriever


def test_pipeline_suggests_code_for_extracted_entity(indexed_client: QdrantClient) -> None:
    # The fake cross-encoder scores by token overlap, so an assessment that
    # repeats the description tokens clears the 0.7 confidence threshold.
    assessment = "unspecified nonsuppurative otitis media bilateral"
    pipeline, _ = _make_pipeline(
        indexed_client, ['["Unspecified nonsuppurative otitis media, bilateral"]']
    )

    suggested = pipeline.run(assessment, patient_sex="female", patient_age=6)

    assert suggested, "expected at least one suggested code"
    top = suggested[0]
    assert top["code"] == "H65.93"
    assert top["confidence"] > 0.7
    assert top["pending_confirmation"] is True
    assert top["source"] == "ICD-10-CM 2026"


def test_pipeline_zero_hit_returns_empty_list(indexed_client: QdrantClient) -> None:
    """No candidate overlaps the assessment -> all confidences below the
    threshold -> graceful degradation (empty list; node emits ZERO_HIT_MESSAGE)."""
    pipeline, _ = _make_pipeline(indexed_client, ['["quantum flux capacitor syndrome"]'])
    assert pipeline.run("zzz qqq xxx") == []
    assert "manually assign" in ZERO_HIT_MESSAGE


def test_pipeline_falls_back_to_raw_assessment_when_extraction_fails(
    indexed_client: QdrantClient,
) -> None:
    """Unparseable LLM output must degrade to querying with the assessment
    text itself, never crash the pipeline."""
    assessment = BY_CODE["I21.9"].to_embedding_text()
    pipeline, retriever = _make_pipeline(indexed_client, ["not json at all"])

    suggested = pipeline.run(assessment)

    assert retriever.calls == 1  # queried with the raw assessment
    assert suggested[0]["code"] == "I21.9"


def test_pipeline_caches_retrieval_per_entity_and_demographics(
    indexed_client: QdrantClient,
) -> None:
    entity_json = '["Acute myocardial infarction, unspecified"]'
    assessment = "acute myocardial infarction, unspecified"
    pipeline, retriever = _make_pipeline(indexed_client, [entity_json] * 3)

    pipeline.run(assessment, patient_sex="male", patient_age=60)
    assert retriever.calls == 1

    # Same entity + same demographics -> served from cache.
    pipeline.run(assessment, patient_sex="male", patient_age=60)
    assert retriever.calls == 1

    # Same entity, different patient -> different filter -> fresh retrieval.
    pipeline.run(assessment, patient_sex="female", patient_age=30)
    assert retriever.calls == 2


def test_pipeline_deduplicates_candidates_across_entities(
    indexed_client: QdrantClient,
) -> None:
    """Two entities hitting the same code must not produce duplicate suggestions."""
    entity_json = json.dumps(
        ["Acute myocardial infarction, unspecified", "myocardial infarction acute"]
    )
    assessment = "acute myocardial infarction unspecified"
    pipeline, _ = _make_pipeline(indexed_client, [entity_json])

    suggested = pipeline.run(assessment)
    codes = [c["code"] for c in suggested]
    assert len(codes) == len(set(codes))


def test_pipeline_rejects_empty_assessment(indexed_client: QdrantClient) -> None:
    pipeline, _ = _make_pipeline(indexed_client, [])
    with pytest.raises(ValueError):
        pipeline.run("")


def test_pipeline_guarantees_each_entity_its_best_code(indexed_client: QdrantClient) -> None:
    """Multi-entity crowding: one entity's high-scoring candidates must not
    push another entity's best code out of the merged top-k entirely —
    measured live on TX001, three extracted conditions produced a top-3 that
    was ALL insomnia codes, dropping the chief complaint (headache).
    Every extracted condition gets at least its best code represented."""
    from types import SimpleNamespace

    def canned_retriever(cands_by_query):
        return SimpleNamespace(
            retrieve=lambda q, patient_sex="unknown", patient_age=None: cands_by_query[q]
        )

    def canned_reranker(conf_by_code):
        def rerank(query, candidates, top_n):
            scored = [{**c, "confidence": conf_by_code[c["code"]]} for c in candidates]
            scored.sort(key=lambda c: c["confidence"], reverse=True)
            return scored[:top_n]

        return SimpleNamespace(rerank=rerank)

    # Entity A dominates with three 0.97+ codes; entity B's best is 0.90.
    cands = {
        "Insomnia": [{"code": f"F51.{i}", "text": "t", "children_codes": [], "score": 1.0,
                      "description": "d", "hierarchy_path": "h"} for i in range(3)],
        "Tension headache": [{"code": "G44.2", "text": "t", "children_codes": [], "score": 1.0,
                              "description": "d", "hierarchy_path": "h"}],
    }
    confs = {"F51.0": 0.99, "F51.1": 0.98, "F51.2": 0.97, "G44.2": 0.90}

    pipeline = RAGPipeline(
        entity_extractor=FakeChatModel([]),  # bypassed via entities=
        retriever=canned_retriever(cands),
        reranker=canned_reranker(confs),
        specificity_checker=SpecificityChecker(client=indexed_client, collection_name=COLLECTION),
        cache=RAGCache(max_size=8),
    )
    suggested = pipeline.run(
        "assessment", entities=["Insomnia", "Tension headache"]
    )
    codes = [s["code"] for s in suggested]
    assert "G44.2" in codes, "entity B's best code must survive the merge"
    assert codes[0] == "F51.0", "ordering stays confidence-descending"
    assert len(codes) == 3

    # But a SUB-threshold best gets no protected slot: with entity B's only
    # code now scoring 0.30, the merge reverts to pure confidence order.
    confs["G44.2"] = 0.30
    pipeline.cache = RAGCache(max_size=8)  # fresh cache; confs changed
    suggested = pipeline.run("assessment", entities=["Insomnia", "Tension headache"])
    assert [s["code"] for s in suggested] == ["F51.0", "F51.1", "F51.2"]


def test_pipeline_accepts_pre_extracted_entities(indexed_client: QdrantClient) -> None:
    """The graph's entity_extraction node runs first; run(entities=...) must
    skip internal extraction (no LLM call) and use the provided entities."""
    pipeline, retriever = _make_pipeline(indexed_client, [])  # NO canned LLM replies
    assessment = "acute myocardial infarction unspecified"

    suggested = pipeline.run(
        assessment, entities=["Acute myocardial infarction, unspecified"]
    )

    assert retriever.calls == 1
    assert suggested[0]["code"] == "I21.9"
