"""Tests for the Gradio UI handlers (Task 9 wiring + Task 16 save/trace/memory)."""

from __future__ import annotations

import pytest

from mednote.ui import app


@pytest.fixture(autouse=True)
def silence_gradio_toasts(monkeypatch: pytest.MonkeyPatch):
    """gr.Warning/Info need a request context; no-op them for handler tests."""
    monkeypatch.setattr(app.gr, "Warning", lambda *a, **k: None)
    monkeypatch.setattr(app.gr, "Info", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point visit memory at a temp SQLite file so UI tests never touch
    the real data/memory.db."""
    from mednote.memory import store

    monkeypatch.setattr(store, "get_db_path", lambda: str(tmp_path / "memory.db"))


SAMPLE_CODES = [
    {
        "code": "I20.9",
        "description": "Angina pectoris, unspecified",
        "source": "ICD-10-CM 2026",
        "confidence": 0.91,
        "specificity_options": [],
    },
    {
        "code": "R11.0",
        "description": "Nausea",
        "source": "ICD-10-CM 2026",
        "confidence": 0.84,
        "specificity_options": [{"code": "R11.10", "description": "Vomiting, unspecified"}],
    },
]

NOTE_WITH_CODES = (
    "### Subjective\nok\n\n### Plan\nrest\n\n### Suggested ICD-10 Codes\n"
    "I20.9 - Angina pectoris (Source: ICD-10-CM 2026) (Pending Physician Confirmation)"
)

EMERGENCY_NOTE = (
    "🚨 URGENT ESCALATION REQUIRED\n\n"
    "Red-flag symptom combination detected: **chest pain radiating to the left arm** "
    "with **diaphoresis** and **nausea** — treat as possible acute coronary syndrome.\n"
    "Recommend immediate in-person emergency evaluation.\n\n"
    "### Subjective\nChest pain for 20 minutes.\n\n### Plan\nEmergency referral."
)


def test_empty_transcript_keeps_empty_state(monkeypatch) -> None:
    empty_update, emergency_update, note_update, codes_update, save_row, save_status, _, last = (
        app.generate_note("   ")
    )
    assert empty_update["visible"] is True
    assert emergency_update["visible"] is False
    assert note_update["visible"] is False
    assert codes_update["visible"] is False
    assert save_row["visible"] is False
    assert last is None


def test_generate_note_renders_note_without_inline_codes_section(monkeypatch) -> None:
    def fake_run_agent(transcript, **kwargs):
        assert kwargs["patient_age"] == app.MOCK_PATIENT_AGE
        assert kwargs["patient_sex"] == app.MOCK_PATIENT_SEX
        return {
            "final_response": NOTE_WITH_CODES,
            "errors": [],
            "suggested_codes": SAMPLE_CODES,
        }

    monkeypatch.setattr(app, "run_agent", fake_run_agent)
    empty_update, emergency_update, note_update, codes_update, save_row, _, _, last = (
        app.generate_note("Doctor: hello\nPatient: hi")
    )

    assert empty_update["visible"] is False
    assert emergency_update["visible"] is False  # routine note: no banner
    assert note_update["visible"] is True
    assert "### Subjective" in note_update["value"]
    # The LLM's inline codes lines are replaced by the structured panel.
    assert "### Suggested ICD-10 Codes" not in note_update["value"]
    assert codes_update["visible"] is True
    # Task 16: the Save button appears, and state stages what it will save.
    assert save_row["visible"] is True
    assert last["patient_id"] == app.MOCK_PATIENT_ID
    assert last["codes"] == SAMPLE_CODES


# ------------------------------------------------------- emergency banner ---


def test_split_escalation_detects_llm_preamble() -> None:
    banner, remainder = app.split_escalation(EMERGENCY_NOTE)
    assert banner is not None
    assert "URGENT ESCALATION" in banner
    assert "acute coronary syndrome" in banner
    # The SOAP note survives intact, without the escalation preamble.
    assert remainder.startswith("### Subjective")
    assert "URGENT ESCALATION" not in remainder


def test_split_escalation_detects_guardrail_prefix() -> None:
    from mednote.agent.prompts import ESCALATION_PROMPT

    note = ESCALATION_PROMPT.format(reason="chest pain radiating to arm") + "\n\n### Subjective\nok"
    banner, remainder = app.split_escalation(note)
    assert banner is not None
    assert "chest pain radiating to arm" in banner
    assert remainder.startswith("### Subjective")


def test_split_escalation_leaves_routine_note_untouched() -> None:
    banner, remainder = app.split_escalation(NOTE_WITH_CODES)
    assert banner is None
    assert remainder == NOTE_WITH_CODES


def test_emergency_banner_renders_risks_bold_and_red() -> None:
    banner, _ = app.split_escalation(EMERGENCY_NOTE)
    html_out = app.render_emergency_banner(banner)

    assert 'class="emergency-banner"' in html_out
    assert 'role="alert"' in html_out
    # Named risks come out as <strong> inside the red banner.
    assert "<strong>chest pain radiating to the left arm</strong>" in html_out
    assert "<strong>diaphoresis</strong>" in html_out
    # No raw markdown emphasis or unescaped user text leaks through.
    assert "**" not in html_out
    # The red styling is pinned in the CSS the app ships.
    assert ".emergency-banner" in app.CSS
    assert "#d93025" in app.CSS or "#b3261e" in app.CSS


def test_generate_note_shows_emergency_banner_for_red_flag_note(monkeypatch) -> None:
    monkeypatch.setattr(
        app,
        "run_agent",
        lambda *a, **k: {
            "final_response": EMERGENCY_NOTE,
            "errors": [],
            "suggested_codes": SAMPLE_CODES,
        },
    )
    _, emergency_update, note_update, codes_update, *_ = app.generate_note("chest pain transcript")

    assert emergency_update["visible"] is True
    assert "URGENT ESCALATION" in emergency_update["value"]
    # The note panel shows the SOAP note only — escalation lives in the banner.
    assert note_update["visible"] is True
    assert "URGENT ESCALATION" not in note_update["value"]
    assert codes_update["visible"] is True


def test_codes_panel_chips_hover_checkbox_and_single_source(monkeypatch) -> None:
    html_out = app.render_code_chips(SAMPLE_CODES)

    # Chips show the CODE text; descriptions live in the hover tooltip.
    assert ">I20.9<" in html_out.replace(" ", "")
    assert 'title="' in html_out
    assert "Angina pectoris, unspecified" in html_out       # inside title attr
    assert html_out.count("Angina pectoris, unspecified") == 1
    # One checkbox per code, for physician confirmation.
    assert html_out.count('type="checkbox"') == len(SAMPLE_CODES)
    assert "Pending Physician Confirmation" in html_out
    # Source cited exactly ONCE, at the end — not per code line.
    assert html_out.count("ICD-10-CM 2026") == 1
    # Specificity options surface in the hover text.
    assert "R11.10" in html_out


def test_codes_panel_empty_list_renders_nothing() -> None:
    assert app.render_code_chips([]) == ""


def test_generate_note_surfaces_error_channel(monkeypatch) -> None:
    monkeypatch.setattr(
        app,
        "run_agent",
        lambda *a, **k: {"final_response": "note", "errors": ["zero hit"], "suggested_codes": []},
    )
    _, _, note_update, codes_update, *_ = app.generate_note("some transcript")
    assert "⚠️ zero hit" in note_update["value"]
    assert codes_update["visible"] is False


def test_generate_note_survives_agent_failure(monkeypatch) -> None:
    def explode(*a, **k):
        raise RuntimeError("qdrant locked")

    monkeypatch.setattr(app, "run_agent", explode)
    empty_update, emergency_update, note_update, codes_update, save_row, _, _, last = (
        app.generate_note("some transcript")
    )

    assert empty_update["visible"] is True   # UI falls back, never crashes
    assert emergency_update["visible"] is False
    assert note_update["visible"] is False
    assert codes_update["visible"] is False
    assert save_row["visible"] is False
    assert last is None


# ------------------------------------------------ Task 16: save / trace / memory ---


FULL_FINAL_STATE = {
    "trace_id": "t-123",
    "intent": "soap",
    "final_response": NOTE_WITH_CODES,
    "draft_note": "RAW DRAFT NOTE",
    "extracted_entities": ["Angina pectoris"],
    "suggested_codes": SAMPLE_CODES,
    "cache_hit": True,
    "guardrail_result": {"passed": True, "is_red_flag": False, "severity": "info", "flags": []},
    "errors": [],
}


def test_generate_note_builds_trace_and_stages_draft(monkeypatch) -> None:
    monkeypatch.setattr(app, "run_agent", lambda *a, **k: dict(FULL_FINAL_STATE))
    *_, trace_update, last = app.generate_note("Doctor: hello\nPatient: hi")

    trace = trace_update["value"]
    assert trace["trace_id"] == "t-123"
    assert trace["intent"] == "soap"
    assert trace["cache_hit"] is True
    assert trace["suggested_codes"] == [
        {"code": "I20.9", "confidence": 0.91},
        {"code": "R11.0", "confidence": 0.84},
    ]
    assert isinstance(trace["latency_ms"], int)
    # The Save button hands tool_execution the RAW draft, not the UI rendering.
    assert last["note"] == "RAW DRAFT NOTE"


def test_generate_note_passes_memory_context(monkeypatch) -> None:
    from mednote.memory.store import MemoryStore

    MemoryStore().save_visit(
        app.MOCK_PATIENT_ID, "2026-07-19", "N_prior", "Prior headache visit.", ["G44.2"]
    )
    seen = {}

    def fake_run_agent(transcript, **kwargs):
        seen.update(kwargs)
        return dict(FULL_FINAL_STATE)

    monkeypatch.setattr(app, "run_agent", fake_run_agent)
    app.generate_note("Doctor: hello\nPatient: hi")
    assert seen["memory_context"] is not None
    assert "Prior headache visit." in seen["memory_context"]["summary"]


def test_save_note_to_ehr_success_banner_and_history_refresh(monkeypatch) -> None:
    def fake_run_agent(user_input, **kwargs):
        assert kwargs["draft_note"] == "RAW DRAFT NOTE"       # staged note reached the agent
        assert kwargs["suggested_codes"] == SAMPLE_CODES
        return {
            "trace_id": "t-save", "intent": "save", "errors": [],
            "tool_result": {"ok": True, "detail": "saved", "note_id": "N_ui123"},
        }

    monkeypatch.setattr(app, "run_agent", fake_run_agent)
    status, trace_update, history = app.save_note_to_ehr(
        {"patient_id": "P-DEMO", "note": "RAW DRAFT NOTE", "codes": SAMPLE_CODES}
    )

    assert status["visible"] is True
    assert "N_ui123" in status["value"] and "success" in status["value"]
    assert trace_update["value"]["intent"] == "save"
    assert "No prior visits found" in history["value"]  # temp store: EHR saved, memory empty


def test_save_note_to_ehr_failure_banner(monkeypatch) -> None:
    monkeypatch.setattr(
        app,
        "run_agent",
        lambda *a, **k: {
            "errors": ["down"],
            "tool_result": {"ok": False, "detail": "the note was NOT saved", "note_id": None},
        },
    )
    status, _, _ = app.save_note_to_ehr({"patient_id": "P-DEMO", "note": "N", "codes": []})
    assert status["visible"] is True
    assert "failure" in status["value"]
    assert "NOT saved" in status["value"]


def test_save_note_to_ehr_without_note_warns_and_does_nothing(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(app, "run_agent", lambda *a, **k: called.append(1))
    status, _, _ = app.save_note_to_ehr(None)
    assert status["visible"] is False
    assert called == []  # guardrail G5: nothing runs without a staged note


def test_render_prior_visits_reads_memory() -> None:
    from mednote.memory.store import MemoryStore

    assert "No prior visits found" in app.render_prior_visits("P-DEMO")
    MemoryStore().save_visit("P-DEMO", "2026-07-19", "N_1", "Tension headache visit.", ["G44.2"])
    text = app.render_prior_visits("P-DEMO")
    assert "Prior visits for patient P-DEMO" in text
    assert "Tension headache visit." in text and "G44.2" in text
