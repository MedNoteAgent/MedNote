"""Tests for the mock EHR (Task 11) and its two tool wrappers (Tasks 11-12).

EHRStore is exercised directly (unit-level) and through the FastAPI app
(REST-level, via TestClient) so both the "realistic REST surface" and the
in-process tool-call path documented in docs/tools.md are covered.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mednote.tools.ehr_api import EHRStore, app
from mednote.tools.get_history import get_patient_history
from mednote.tools.save_note import save_note


@pytest.fixture()
def store(tmp_path) -> EHRStore:
    return EHRStore(str(tmp_path / "ehr_store.json"))


# ------------------------------------------------------------- EHRStore ---


def test_save_note_returns_status_saved_with_generated_id(store: EHRStore) -> None:
    result = store.save_note("P001", "### Subjective\nHeadache.", ["G44.2"])
    assert result["status"] == "saved"
    assert result["note_id"].startswith("N_")
    assert result["patient_id"] == "P001"
    assert "timestamp" in result


def test_save_note_missing_patient_id_is_an_error(store: EHRStore) -> None:
    result = store.save_note("", "some note")
    assert result == {"status": "error", "message": "Patient ID is required"}


def test_save_note_missing_note_text_is_an_error(store: EHRStore) -> None:
    result = store.save_note("P001", "   ")
    assert result == {"status": "error", "message": "Note text is required"}


def test_get_history_no_prior_visits(store: EHRStore) -> None:
    result = store.get_history("P999")
    assert result == {
        "status": "no_history",
        "patient_id": "P999",
        "message": "No prior visits found",
    }


def test_get_history_returns_visits_newest_first(store: EHRStore) -> None:
    first = store.save_note("P001", "Visit one note")
    second = store.save_note("P001", "Visit two note")

    result = store.get_history("P001")
    assert result["status"] == "found"
    ids = [v["note_id"] for v in result["visits"]]
    assert ids[0] == second["note_id"]
    assert ids[1] == first["note_id"]


def test_get_history_only_returns_the_requested_patient(store: EHRStore) -> None:
    store.save_note("P001", "note for P001")
    store.save_note("P002", "note for P002")

    result = store.get_history("P001")
    assert len(result["visits"]) == 1
    assert result["visits"][0]["patient_id"] == "P001"


def test_get_history_missing_patient_id_is_an_error(store: EHRStore) -> None:
    assert store.get_history("") == {"status": "error", "message": "Patient ID is required"}


def test_store_persists_across_instances(tmp_path) -> None:
    """The JSON file, not the object, is the source of truth."""
    path = str(tmp_path / "ehr_store.json")
    EHRStore(path).save_note("P001", "first note")

    reopened = EHRStore(path)
    result = reopened.get_history("P001")
    assert result["status"] == "found"
    assert result["visits"][0]["note"] == "first note"


# --------------------------------------------------------------- FastAPI ---


def test_rest_post_notes_then_get_history(monkeypatch, store: EHRStore) -> None:
    import mednote.tools.ehr_api as ehr_api_module

    monkeypatch.setattr(ehr_api_module, "get_ehr_store", lambda: store)
    client = TestClient(app)

    post_resp = client.post(
        "/notes", json={"patient_id": "P001", "note": "SOAP note text", "icd_codes": ["G44.2"]}
    )
    assert post_resp.status_code == 200
    assert post_resp.json()["status"] == "saved"

    get_resp = client.get("/patients/P001/history")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["status"] == "found"
    assert body["visits"][0]["note"] == "SOAP note text"


def test_rest_post_notes_missing_note_returns_400(monkeypatch, store: EHRStore) -> None:
    import mednote.tools.ehr_api as ehr_api_module

    monkeypatch.setattr(ehr_api_module, "get_ehr_store", lambda: store)
    client = TestClient(app)

    resp = client.post("/notes", json={"patient_id": "P001", "note": ""})
    assert resp.status_code == 400
    assert resp.json()["status"] == "error"


# ------------------------------------------------------ LangChain tools ---


def test_save_note_tool_invokes_the_ehr_store(monkeypatch, store: EHRStore) -> None:
    import mednote.tools.save_note as save_note_module

    monkeypatch.setattr(save_note_module, "get_ehr_store", lambda: store)
    result = save_note.invoke({"patient_id": "P001", "note": "a note", "icd_codes": ["J02.9"]})

    assert result["status"] == "saved"
    assert store.get_history("P001")["visits"][0]["icd_codes"] == ["J02.9"]


def test_get_patient_history_tool_invokes_the_ehr_store(monkeypatch, store: EHRStore) -> None:
    import mednote.tools.get_history as get_history_module

    monkeypatch.setattr(get_history_module, "get_ehr_store", lambda: store)
    store.save_note("P001", "a prior note")

    result = get_patient_history.invoke({"patient_id": "P001"})
    assert result["status"] == "found"
    assert result["visits"][0]["note"] == "a prior note"


def test_get_patient_history_tool_reports_no_history_for_new_patient(
    monkeypatch, store: EHRStore
) -> None:
    import mednote.tools.get_history as get_history_module

    monkeypatch.setattr(get_history_module, "get_ehr_store", lambda: store)
    result = get_patient_history.invoke({"patient_id": "P999"})
    assert result["status"] == "no_history"
