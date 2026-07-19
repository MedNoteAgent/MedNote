"""Tests for the EHR client + LangChain tool wrappers (Tasks 11-12).

The tools run against the real FastAPI app through an injected TestClient —
full HTTP round-trip, no network, temp-file store.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from mednote.tools import ehr_api, ehr_client, get_patient_history, save_note


@pytest.fixture()
def ehr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ehr_api, "get_store_path", lambda: tmp_path / "ehr_store.json")
    monkeypatch.setattr(ehr_client, "get_client", lambda: TestClient(ehr_api.app))


def test_save_note_tool_round_trip(ehr) -> None:
    result = save_note.invoke(
        {"patient_id": "P001", "note": "SOAP body", "icd_codes": ["G44.2"]}
    )
    assert result["status"] == "saved"
    assert result["note_id"].startswith("N_")


def test_get_history_tool_round_trip(ehr) -> None:
    assert get_patient_history.invoke({"patient_id": "P001"})["status"] == "no_history"

    save_note.invoke({"patient_id": "P001", "note": "SOAP body"})
    result = get_patient_history.invoke({"patient_id": "P001"})
    assert result["status"] == "found"
    assert len(result["visits"]) == 1


def test_tools_expose_llm_facing_schemas() -> None:
    """bind_tools() serves these to the LLM — names, docs, and args must exist."""
    assert save_note.name == "save_note"
    assert get_patient_history.name == "get_patient_history"
    assert "physician" in save_note.description  # confirmation rule reaches the LLM
    assert set(save_note.args) == {"patient_id", "note", "icd_codes"}
    assert set(get_patient_history.args) == {"patient_id"}


def test_client_raises_ehr_api_error_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ehr_client,
        "get_client",
        lambda: httpx.Client(base_url="http://127.0.0.1:59999", timeout=0.2),
    )
    with pytest.raises(ehr_client.EhrApiError):
        ehr_client.post_note("P001", "x")


def test_demographics_fetch(ehr) -> None:
    result = ehr_client.fetch_demographics("P005")
    assert result["status"] == "found"
    assert result["patient"]["sex"] == "male"
