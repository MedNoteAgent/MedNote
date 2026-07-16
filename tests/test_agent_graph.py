"""Tests for the LangGraph wiring (Task 8).

Full graph invocations with the heavy services faked via the nodes module's
factory seams (get_rag_pipeline / get_note_llm) — no model downloads, no API
calls. The live DoD check (real transcript -> real SOAP note) runs separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mednote.agent import nodes
from mednote.agent.graph import build_graph, run_agent
from mednote.agent.nodes import INSUFFICIENT_INPUT_MESSAGE, parse_input
from mednote.agent.prompts import REFUSAL_PROMPT
from mednote.agent.state import make_initial_state
from mednote.rag.pipeline import ZERO_HIT_MESSAGE

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = json.loads(
    (REPO_ROOT / "data" / "transcripts" / "synthetic_transcripts.json").read_text(
        encoding="utf-8"
    )
)
BY_ID = {t["transcript_id"]: t for t in TRANSCRIPTS}

G442 = {
    "code": "G44.2",
    "description": "Tension-type headache",
    "hierarchy_path": "Nervous system -> Episodic disorders",
    "source": "ICD-10-CM 2026",
    "confidence": 0.95,
    "parent_code": "G44",
    "specificity_options": [],
    "pending_confirmation": True,
}

CANNED_NOTE = (
    "### Subjective\nHeadache for 3 days.\n\n### Objective\nBP 130/85.\n\n"
    "### Assessment\nPossible tension-type headache — FOR PHYSICIAN REVIEW.\n\n"
    "### Plan\nFollow-up in 2 weeks.\n\n### Suggested ICD-10 Codes\n"
    "G44.2 - Tension-type headache (Source: ICD-10-CM 2026) "
    "(Pending Physician Confirmation)"
)


class FakeRAGPipeline:
    """Stands in for RAGPipeline; returns canned codes, records calls."""

    def __init__(self, codes: list[dict], entities: list[str] | None = None):
        self._codes = codes
        self.cache = SimpleNamespace(hits=0)
        self.entity_extractor = SimpleNamespace(
            extract=lambda text: entities or ["Tension-type headache"]
        )
        self.run_calls: list[dict] = []

    def run(self, assessment_text, patient_sex="unknown", patient_age=None, entities=None):
        self.run_calls.append(
            {"sex": patient_sex, "age": patient_age, "entities": entities}
        )
        return self._codes


class FakeNoteLLM:
    def __init__(self, reply: str = CANNED_NOTE):
        self._reply = reply
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self._reply)


class ExplodingLLM:
    def invoke(self, messages):  # pragma: no cover - failure is the assertion
        raise AssertionError("note LLM must not be called on this path")


@pytest.fixture()
def fakes(monkeypatch: pytest.MonkeyPatch):
    pipeline = FakeRAGPipeline([G442])
    llm = FakeNoteLLM()
    monkeypatch.setattr(nodes, "get_rag_pipeline", lambda: pipeline)
    monkeypatch.setattr(nodes, "get_note_llm", lambda: llm)
    return pipeline, llm


# ------------------------------------------------------------- parse_input ---


def test_parse_input_routes_all_five_intents() -> None:
    cases = {
        "TX001": "soap",        # dialogue transcript
        "TX002": "icd_lookup",  # "What ICD-10 code fits..."
        "TX003": "save",        # "Save this note..."
        "TX004": "history",     # "...last visit?"
        "TX006": "refuse",      # "Diagnose this patient..."
    }
    for tx_id, expected in cases.items():
        result = parse_input(make_initial_state(BY_ID[tx_id]["transcript"], "t"))
        assert result["intent"] == expected, tx_id


def test_parse_input_dialogue_beats_keywords() -> None:
    """TX005 contains 'any HISTORY of heart problems' — the plan's bare
    keyword sketch would misroute the red-flag encounter to intent=history."""
    result = parse_input(make_initial_state(BY_ID["TX005"]["transcript"], "t"))
    assert result["intent"] == "soap"


def test_every_dataset_row_routes_to_its_expected_intent() -> None:
    for entry in TRANSCRIPTS:
        result = parse_input(make_initial_state(entry["transcript"], "t"))
        assert result["intent"] == entry["expected_intent"], entry["transcript_id"]


# ---------------------------------------------------------- graph, per path ---


def test_soap_round_trip_with_pending_confirmation(fakes) -> None:
    pipeline, llm = fakes
    final = run_agent(
        BY_ID["TX001"]["transcript"], patient_age=34, patient_sex="female"
    )

    assert final["intent"] == "soap"
    assert final["suggested_codes"] == [G442]
    assert llm.calls == 1
    for header in ("### Subjective", "### Objective", "### Assessment",
                   "### Plan", "### Suggested ICD-10 Codes"):
        assert header in final["final_response"]
    assert "(Pending Physician Confirmation)" in final["final_response"]
    # Demographics flowed into the retrieval filter.
    assert pipeline.run_calls[0]["sex"] == "female"
    assert pipeline.run_calls[0]["age"] == 34
    # Entities were extracted once, by the entity_extraction node.
    assert pipeline.run_calls[0]["entities"] == ["Tension-type headache"]


def test_icd_lookup_skips_note_generation(fakes, monkeypatch) -> None:
    monkeypatch.setattr(nodes, "get_note_llm", lambda: ExplodingLLM())
    final = run_agent(BY_ID["TX002"]["transcript"])

    assert final["intent"] == "icd_lookup"
    assert "G44.2" in final["final_response"]
    assert "(Source: ICD-10-CM 2026)" in final["final_response"]
    assert "(Pending Physician Confirmation)" in final["final_response"]


def test_refuse_path_returns_refusal_prompt(fakes) -> None:
    final = run_agent(BY_ID["TX006"]["transcript"])
    assert final["final_response"] == REFUSAL_PROMPT


def test_save_path_reports_stub_tool_honestly(fakes) -> None:
    final = run_agent(BY_ID["TX003"]["transcript"])
    assert final["tool_result"]["ok"] is False
    assert "NOT saved" in final["final_response"]


def test_history_path_reports_stub_memory(fakes) -> None:
    final = run_agent(BY_ID["TX004"]["transcript"], patient_id="P001")
    assert "No prior visit records" in final["final_response"]


def test_zero_hit_accumulates_error_and_still_drafts(monkeypatch) -> None:
    pipeline = FakeRAGPipeline(codes=[])
    monkeypatch.setattr(nodes, "get_rag_pipeline", lambda: pipeline)
    monkeypatch.setattr(nodes, "get_note_llm", lambda: FakeNoteLLM())

    final = run_agent(BY_ID["TX001"]["transcript"])
    assert ZERO_HIT_MESSAGE in final["errors"]
    assert final["suggested_codes"] == []
    assert "### Subjective" in final["final_response"]  # note still drafted


def test_short_transcript_degrades_without_llm_call(fakes, monkeypatch) -> None:
    monkeypatch.setattr(nodes, "get_note_llm", lambda: ExplodingLLM())
    final = run_agent(BY_ID["TX018"]["transcript"])  # below min_transcript_words

    assert final["final_response"] == INSUFFICIENT_INPUT_MESSAGE
    assert INSUFFICIENT_INPUT_MESSAGE in final["errors"]


def test_extraction_failure_falls_back_to_raw_transcript(monkeypatch) -> None:
    pipeline = FakeRAGPipeline([G442])

    def explode(text):
        raise ValueError("unparseable")

    pipeline.entity_extractor = SimpleNamespace(extract=explode)
    monkeypatch.setattr(nodes, "get_rag_pipeline", lambda: pipeline)
    monkeypatch.setattr(nodes, "get_note_llm", lambda: FakeNoteLLM())

    final = run_agent(BY_ID["TX001"]["transcript"])
    assert final["extracted_entities"] == [BY_ID["TX001"]["transcript"]]
    assert final["suggested_codes"] == [G442]


def test_graph_compiles_and_state_seeds_minimal() -> None:
    assert build_graph() is not None
    state = make_initial_state("hello", "trace-1")
    assert state == {"user_input": "hello", "trace_id": "trace-1", "errors": []}


# ------------------------------------------------- service factory threading ---


def test_get_rag_pipeline_builds_once_under_concurrent_first_calls(monkeypatch) -> None:
    """UI warm-up thread + a Generate click must never build two pipelines:
    the second embedded QdrantClient on the same path dies with AlreadyLocked
    (portalocker) inside the very same process."""
    import threading
    import time

    build_calls = []
    build_started = threading.Event()

    def slow_build():
        build_calls.append(1)
        build_started.set()
        time.sleep(0.05)  # hold the build window open for the second thread
        return FakeRAGPipeline([G442])

    monkeypatch.setattr(nodes, "_rag_pipeline", None)
    monkeypatch.setattr(nodes, "_build_rag_pipeline", slow_build)

    results: list = []
    threads = [
        threading.Thread(target=lambda: results.append(nodes.get_rag_pipeline()))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(build_calls) == 1, "concurrent first calls must share one build"
    assert results[0] is results[1]
