"""Tests for the MCP server (Task 13): both EHR tools exposed via FastMCP.

Calls FastMCP.call_tool() directly against the in-process server instance —
this exercises the real MCP tool-dispatch path (schema validation, docstring
-> description, argument binding) without needing a stdio subprocess.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from mednote.mcp.server import mcp_server
from mednote.tools.ehr_api import EHRStore


@pytest.fixture()
def store(tmp_path, monkeypatch) -> EHRStore:
    import mednote.tools.get_history as get_history_module
    import mednote.tools.save_note as save_note_module

    ehr_store = EHRStore(str(tmp_path / "ehr_store.json"))
    monkeypatch.setattr(save_note_module, "get_ehr_store", lambda: ehr_store)
    monkeypatch.setattr(get_history_module, "get_ehr_store", lambda: ehr_store)
    return ehr_store


def _call(name: str, arguments: dict) -> dict:
    """Run an MCP tool call and unwrap its structured content to a plain dict."""
    result = asyncio.run(mcp_server.call_tool(name, arguments))
    # FastMCP >=? may return (content_blocks, structured_dict) or just one of
    # the two depending on version; normalize both shapes to a dict.
    if isinstance(result, tuple):
        _content, structured = result
        return structured
    if isinstance(result, dict):
        return result
    (block,) = result
    return json.loads(block.text)


def test_mcp_lists_both_tools() -> None:
    tools = asyncio.run(mcp_server.list_tools())
    names = {t.name for t in tools}
    assert {"save_note", "get_patient_history"} <= names


def test_mcp_save_note_round_trip(store: EHRStore) -> None:
    result = _call(
        "save_note",
        {"patient_id": "P001", "note": "### Subjective\nHeadache.", "icd_codes": ["G44.2"]},
    )
    assert result["status"] == "saved"
    assert result["note_id"].startswith("N_")

    # The agent's next call uses the tool's own result — full round trip.
    history = _call("get_patient_history", {"patient_id": "P001"})
    assert history["status"] == "found"
    assert history["visits"][0]["note_id"] == result["note_id"]


def test_mcp_get_patient_history_no_history(store: EHRStore) -> None:
    result = _call("get_patient_history", {"patient_id": "P999"})
    assert result["status"] == "no_history"


def test_mcp_save_note_missing_patient_id_reports_error(store: EHRStore) -> None:
    result = _call("save_note", {"patient_id": "", "note": "some note"})
    assert result["status"] == "error"
    assert "Patient ID" in result["message"]
