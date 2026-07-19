"""Tests for the mock EHR FastAPI server (Task 11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mednote.tools import ehr_api

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(ehr_api, "get_store_path", lambda: tmp_path / "ehr_store.json")
    return TestClient(ehr_api.app)


# ------------------------------------------------------------- POST /notes ---


def test_save_note_returns_note_id(client: TestClient) -> None:
    resp = client.post(
        "/notes",
        json={"patient_id": "P001", "note": "### Subjective\n...", "icd_codes": ["G44.2"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "saved"
    assert body["note_id"].startswith("N_")
    assert body["patient_id"] == "P001"
    assert body["timestamp"]


def test_save_note_validates_required_fields(client: TestClient) -> None:
    assert client.post("/notes", json={"note": "x"}).status_code == 422
    assert client.post("/notes", json={"patient_id": "P001"}).status_code == 422
    assert client.post("/notes", json={"patient_id": "P001", "note": ""}).status_code == 422


def test_save_note_unknown_patient_is_404_error_envelope(client: TestClient) -> None:
    resp = client.post("/notes", json={"patient_id": "NOPE", "note": "x"})
    assert resp.status_code == 404
    body = resp.json()
    assert body["status"] == "error"
    assert "NOPE" in body["message"]


# ------------------------------------------------- GET /patients/{id}/history ---


def test_history_roundtrip(client: TestClient) -> None:
    assert client.get("/patients/P001/history").json()["status"] == "no_history"

    client.post("/notes", json={"patient_id": "P001", "note": "visit 1", "icd_codes": ["G44.2"]})
    client.post("/notes", json={"patient_id": "P002", "note": "other patient"})

    body = client.get("/patients/P001/history").json()
    assert body["status"] == "found"
    assert len(body["visits"]) == 1  # P002's note must not leak in
    assert body["visits"][0]["icd_codes"] == ["G44.2"]
    assert body["visits"][0]["note"] == "visit 1"


def test_history_unknown_patient_is_404(client: TestClient) -> None:
    resp = client.get("/patients/NOPE/history")
    assert resp.status_code == 404
    assert resp.json()["status"] == "error"


# --------------------------------------------------------- GET /patients/{id} ---


def test_demographics_serve_dataset_patients(client: TestClient) -> None:
    body = client.get("/patients/P005").json()
    assert body["status"] == "found"
    assert body["patient"]["age"] == 4
    assert body["patient"]["sex"] == "male"


def test_demographics_unknown_patient_is_404(client: TestClient) -> None:
    assert client.get("/patients/NOPE").status_code == 404


# ----------------------------------------------------------------- seed data ---


def test_seed_covers_every_dataset_patient_with_matching_demographics() -> None:
    """Demographics must line up with the Task 4 eval labels exactly."""
    transcripts = json.loads(
        (REPO_ROOT / "data" / "transcripts" / "synthetic_transcripts.json").read_text(
            encoding="utf-8"
        )
    )
    for entry in transcripts:
        patient = ehr_api.SEED_PATIENTS.get(entry["patient_id"])
        assert patient is not None, entry["patient_id"]
        assert patient["age"] == entry["patient_age"], entry["patient_id"]
        assert patient["sex"] == entry["patient_sex"], entry["patient_id"]
