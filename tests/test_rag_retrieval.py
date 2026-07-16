"""Tests for Task 7 retrieval components: hybrid retriever, reranker,
specificity checker.

All tests run against an in-memory Qdrant populated through the real
ICD10Indexer with deterministic fakes (tests/fakes.py) — no model downloads.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from qdrant_client import QdrantClient

from tests.fakes import FakeCrossEncoder, FakeEmbedder, FakeSparseEncoder, sigmoid

from mednote.rag.etl.parser import ICD10Code
from mednote.rag.indexer import ICD10Indexer
from mednote.rag.reranker import ClinicalReranker
from mednote.rag.retriever import HybridRetriever
from mednote.rag.specificity import SpecificityChecker

CODES = [
    ICD10Code(
        code="I21.9",
        description="Acute myocardial infarction, unspecified",
        hierarchy_path="Circulatory -> Ischemic heart diseases",
        chapter="Diseases of the circulatory system",
        chapter_code="9",
        index_synonyms=["Infarct, infarction, myocardium"],
    ),
    ICD10Code(
        code="J44.1",
        description="Chronic obstructive pulmonary disease with acute exacerbation",
        hierarchy_path="Respiratory -> Chronic lower respiratory diseases",
        chapter="Diseases of the respiratory system",
        chapter_code="10",
        inclusion_terms=["COPD with acute exacerbation"],
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
        code="P22.0",
        description="Respiratory distress syndrome of newborn",
        hierarchy_path="Perinatal -> Respiratory",
        chapter="Certain conditions originating in the perinatal period",
        chapter_code="16",
        max_age_days=28,
    ),
    ICD10Code(
        code="H65.9",
        description="Unspecified nonsuppurative otitis media",
        hierarchy_path="Ear -> Diseases of middle ear",
        chapter="Diseases of the ear and mastoid process",
        chapter_code="8",
        children_codes=["H65.91", "H65.92", "H65.93"],
    ),
    ICD10Code(
        code="H65.91",
        description="Unspecified nonsuppurative otitis media, right ear",
        hierarchy_path="Ear -> Diseases of middle ear -> Unspecified nonsuppurative otitis media",
        chapter="Diseases of the ear and mastoid process",
        chapter_code="8",
        parent_code="H65.9",
    ),
    ICD10Code(
        code="H65.93",
        description="Unspecified nonsuppurative otitis media, bilateral",
        hierarchy_path="Ear -> Diseases of middle ear -> Unspecified nonsuppurative otitis media",
        chapter="Diseases of the ear and mastoid process",
        chapter_code="8",
        parent_code="H65.9",
    ),
]

BY_CODE = {c.code: c for c in CODES}
COLLECTION = "test_retrieval"


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> QdrantClient:
    """One indexed in-memory collection shared across this module's tests."""
    client = QdrantClient(":memory:")
    jsonl = tmp_path_factory.mktemp("data") / "codes.jsonl"
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


@pytest.fixture()
def retriever(client: QdrantClient) -> HybridRetriever:
    return HybridRetriever(
        client=client,
        embedder=FakeEmbedder(),
        sparse_encoder=FakeSparseEncoder(),
        collection_name=COLLECTION,
    )


# -------------------------------------------------------------- retriever ---


def test_retrieve_ranks_exact_dense_match_first(retriever: HybridRetriever) -> None:
    candidates = retriever.retrieve(BY_CODE["I21.9"].to_embedding_text())
    assert candidates[0]["code"] == "I21.9"
    # Fused score is attached and sorted descending.
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_finds_exact_acronym_via_sparse(retriever: HybridRetriever) -> None:
    """'COPD' appears in no description, only in J44.1's inclusion term —
    only the sparse half of the hybrid can find it."""
    candidates = retriever.retrieve("COPD")
    assert candidates[0]["code"] == "J44.1"


def test_retrieve_hard_filters_sex_specific_codes(retriever: HybridRetriever) -> None:
    query = BY_CODE["O80"].to_embedding_text()

    for_female = [c["code"] for c in retriever.retrieve(query, patient_sex="female")]
    assert "O80" in for_female

    for_male = [c["code"] for c in retriever.retrieve(query, patient_sex="male")]
    assert "O80" not in for_male


def test_retrieve_unknown_sex_skips_the_sex_filter(retriever: HybridRetriever) -> None:
    """Never exclude on information we don't have."""
    query = BY_CODE["O80"].to_embedding_text()
    codes = [c["code"] for c in retriever.retrieve(query, patient_sex="unknown")]
    assert "O80" in codes


def test_retrieve_hard_filters_perinatal_codes_by_age(
    retriever: HybridRetriever,
) -> None:
    query = BY_CODE["P22.0"].to_embedding_text()

    for_newborn = [c["code"] for c in retriever.retrieve(query, patient_age=0)]
    assert "P22.0" in for_newborn

    for_adult = [c["code"] for c in retriever.retrieve(query, patient_age=45)]
    assert "P22.0" not in for_adult
    # Age-unrestricted codes are still retrievable for the adult.
    assert for_adult


def test_retrieve_rejects_blank_query(retriever: HybridRetriever) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_retrieve_respects_top_k(retriever: HybridRetriever) -> None:
    candidates = retriever.retrieve("otitis media", top_k=2)
    assert len(candidates) <= 2


# --------------------------------------------------------------- reranker ---


def test_reranker_orders_by_cross_encoder_score() -> None:
    reranker = ClinicalReranker(model=FakeCrossEncoder())
    candidates = [
        {"code": "I21.9", "text": "Acute myocardial infarction, unspecified"},
        {"code": "H65.93", "text": "otitis media bilateral ear infection"},
    ]
    top = reranker.rerank("ear infection in both ears", candidates, top_n=2)
    assert [c["code"] for c in top] == ["H65.93", "I21.9"]


def test_reranker_attaches_sigmoid_confidence() -> None:
    reranker = ClinicalReranker(model=FakeCrossEncoder())
    candidates = [{"code": "X", "text": "alpha beta"}]
    top = reranker.rerank("alpha beta", candidates, top_n=1)
    expected = sigmoid(FakeCrossEncoder._logit("alpha beta", "alpha beta"))
    assert top[0]["confidence"] == pytest.approx(expected)
    # Inputs are not mutated (immutability rule).
    assert "confidence" not in candidates[0]


def test_reranker_truncates_to_top_n() -> None:
    reranker = ClinicalReranker(model=FakeCrossEncoder())
    candidates = [{"code": str(i), "text": f"token{i}"} for i in range(10)]
    assert len(reranker.rerank("query", candidates, top_n=3)) == 3


def test_reranker_empty_candidates_returns_empty() -> None:
    reranker = ClinicalReranker(model=FakeCrossEncoder())
    assert reranker.rerank("query", [], top_n=3) == []


# ------------------------------------------------------------- specificity ---


def test_specificity_expands_parent_code_with_children(client: QdrantClient) -> None:
    checker = SpecificityChecker(client=client, collection_name=COLLECTION)
    suggested = checker.check_and_expand(
        [
            {
                "code": "H65.9",
                "description": "Unspecified nonsuppurative otitis media",
                "hierarchy_path": "Ear -> Diseases of middle ear",
                "children_codes": ["H65.91", "H65.92", "H65.93"],
                "confidence": 0.9,
            }
        ]
    )

    assert len(suggested) == 1
    code = suggested[0]
    assert code["code"] == "H65.9"
    assert code["confidence"] == 0.9
    assert code["pending_confirmation"] is True

    # Children present in the index (H65.91, H65.93) become options;
    # H65.92 was never indexed and must simply be absent, not invented.
    option_codes = {o["code"] for o in code["specificity_options"]}
    assert option_codes == {"H65.91", "H65.93"}
    by_option = {o["code"]: o for o in code["specificity_options"]}
    assert "right ear" in by_option["H65.91"]["description"]


def test_specificity_leaf_code_gets_no_options(client: QdrantClient) -> None:
    checker = SpecificityChecker(client=client, collection_name=COLLECTION)
    suggested = checker.check_and_expand(
        [{"code": "I21.9", "description": "Acute MI", "children_codes": [], "confidence": 0.8}]
    )
    assert suggested[0]["specificity_options"] == []
