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


class FakeToolLLM:
    """Stands in for the tool-bound LLM: returns canned tool_calls."""

    def __init__(self, tool_calls: list[dict], content: str = ""):
        self._tool_calls = tool_calls
        self._content = content
        self.messages = None  # records the prompt, for asserts

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self._content, tool_calls=self._tool_calls)


@pytest.fixture()
def ehr(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Real EHR app on a temp store, injected into the tools' HTTP client."""
    from fastapi.testclient import TestClient

    from mednote.tools import ehr_api, ehr_client

    monkeypatch.setattr(ehr_api, "get_store_path", lambda: tmp_path / "ehr_store.json")
    monkeypatch.setattr(ehr_client, "get_client", lambda: TestClient(ehr_api.app))


@pytest.fixture()
def memory(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Visit-memory store on a temp SQLite file."""
    from mednote.memory import store

    monkeypatch.setattr(store, "get_db_path", lambda: str(tmp_path / "memory.db"))


def test_save_path_executes_tool_and_returns_note_id(fakes, ehr, memory, monkeypatch) -> None:
    tool_llm = FakeToolLLM(
        [{"name": "save_note",
          "args": {"patient_id": "P001", "note": "SOAP body", "icd_codes": ["G44.2"]},
          "id": "call_1", "type": "tool_call"}]
    )
    monkeypatch.setattr(nodes, "get_tool_llm", lambda: tool_llm)

    final = run_agent(
        BY_ID["TX003"]["transcript"],
        patient_id="P001",
        draft_note="STAGED-DRAFT-FROM-UI",   # Task 16: Save button passes it via state
    )
    assert final["tool_result"]["ok"] is True
    assert final["tool_result"]["note_id"].startswith("N_")
    assert final["tool_result"]["note_id"] in final["final_response"]
    # The request context handed to the LLM carried the patient ID and draft.
    assert "P001" in str(tool_llm.messages)
    assert "STAGED-DRAFT-FROM-UI" in str(tool_llm.messages)

    # Task 15: the successful save was mirrored into visit memory.
    from mednote.memory.store import MemoryStore

    visits = MemoryStore().get_history("P001")
    assert len(visits) == 1
    assert visits[0]["note_id"] == final["tool_result"]["note_id"]
    assert visits[0]["icd_codes"] == ["G44.2"]


def test_save_path_degrades_when_ehr_down(fakes, monkeypatch) -> None:
    import httpx

    from mednote.tools import ehr_client

    tool_llm = FakeToolLLM(
        [{"name": "save_note", "args": {"patient_id": "P001", "note": "SOAP body"},
          "id": "call_1", "type": "tool_call"}]
    )
    monkeypatch.setattr(nodes, "get_tool_llm", lambda: tool_llm)
    monkeypatch.setattr(
        ehr_client,
        "get_client",
        lambda: httpx.Client(base_url="http://127.0.0.1:59999", timeout=0.2),
    )

    final = run_agent(BY_ID["TX003"]["transcript"], patient_id="P001")
    assert final["tool_result"]["ok"] is False
    assert "NOT saved" in final["final_response"]
    assert final["errors"]


def test_save_path_without_tool_call_reports_whats_missing(fakes, monkeypatch) -> None:
    tool_llm = FakeToolLLM([], content="No draft note was provided to save.")
    monkeypatch.setattr(nodes, "get_tool_llm", lambda: tool_llm)

    final = run_agent(BY_ID["TX003"]["transcript"], patient_id="P001")
    assert final["tool_result"]["ok"] is False
    assert final["final_response"] == "No draft note was provided to save."


def test_context_extraction_pulls_demographics_from_ehr(ehr) -> None:
    result = nodes.context_extraction({"user_input": "x", "patient_id": "P005"})
    assert result == {"patient_sex": "male", "patient_age": 4}


def test_context_extraction_caller_demographics_win(ehr) -> None:
    result = nodes.context_extraction(
        {"user_input": "x", "patient_id": "P005", "patient_sex": "male", "patient_age": 4}
    )
    assert result == {}


def test_context_extraction_degrades_when_ehr_down(monkeypatch) -> None:
    import httpx

    from mednote.tools import ehr_client

    monkeypatch.setattr(
        ehr_client,
        "get_client",
        lambda: httpx.Client(base_url="http://127.0.0.1:59999", timeout=0.2),
    )
    result = nodes.context_extraction({"user_input": "x", "patient_id": "P001"})
    assert result == {"patient_sex": "unknown"}


def test_history_path_empty_memory_says_no_prior_visits(fakes, memory) -> None:
    final = run_agent(BY_ID["TX004"]["transcript"], patient_id="P001")
    assert "No prior visits found for patient P001" in final["final_response"]


def test_history_path_recalls_saved_visit(fakes, memory) -> None:
    """Task 15 DoD: a visit saved earlier surfaces on the history intent."""
    from mednote.memory.store import MemoryStore

    MemoryStore().save_visit(
        "P001", "2026-07-18", "N_prior1",
        "Possible tension-type headache — physician review.", ["G44.2"],
    )

    final = run_agent(BY_ID["TX004"]["transcript"], patient_id="P001")
    assert "Prior visits for patient P001" in final["final_response"]
    assert "tension-type headache" in final["final_response"]
    assert "N_prior1" in final["final_response"]
    assert "G44.2" in final["final_response"]
    assert final["memory_context"]["prior_visits"][0]["note_id"] == "N_prior1"


def test_history_path_without_patient_id_degrades(fakes, memory) -> None:
    final = run_agent(BY_ID["TX004"]["transcript"])
    assert "No patient ID provided" in final["final_response"]


def test_note_generation_injects_memory_context(fakes, monkeypatch) -> None:
    """Task 15: prior-visit context reaches the SOAP prompt — framed as
    background only — and is absent when there is no memory."""

    class RecordingLLM:
        def __init__(self):
            self.messages = None

        def invoke(self, messages):
            self.messages = messages
            return SimpleNamespace(content=CANNED_NOTE)

    llm = RecordingLLM()
    monkeypatch.setattr(nodes, "get_note_llm", lambda: llm)

    base_state = {"transcript": "Patient reports headache.", "errors": []}
    nodes.note_generation({**base_state, "memory_context": {
        "patient_id": "P001",
        "prior_visits": [{"note_id": "N_prior1"}],
        "summary": "- 2026-07-18: Possible tension-type headache.",
    }})
    prompt = str(llm.messages)
    assert "Possible tension-type headache" in prompt
    assert "continuity" in prompt  # background-only framing, not new findings

    nodes.note_generation(base_state)  # no memory -> no block
    assert "continuity" not in str(llm.messages)


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


def test_close_rag_pipeline_releases_and_is_idempotent(monkeypatch) -> None:
    """The notebooks' §'release the Qdrant lock' entry point (no private pokes)."""
    closed = []
    fake = SimpleNamespace(
        retriever=SimpleNamespace(client=SimpleNamespace(close=lambda: closed.append(1)))
    )
    monkeypatch.setattr(nodes, "_rag_pipeline", fake)

    assert nodes.close_rag_pipeline() is True
    assert closed == [1]
    assert nodes._rag_pipeline is None
    assert nodes.close_rag_pipeline() is False  # nothing left to close
