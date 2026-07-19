"""Tests for the MCP server surface (Task 13).

The MCP tools are one-line delegates to ehr_client (covered by test_tools);
these tests pin the MCP contract: both tools registered, correct schemas,
and a call_tool round-trip through the real store.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mednote.mcp import server
from mednote.tools import ehr_api, ehr_client


def test_registers_both_ehr_tools() -> None:
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert set(tools) == {"save_note", "get_patient_history"}
    for tool in tools.values():
        assert tool.description  # LLM-facing docs must not be empty
    assert "patient_id" in tools["save_note"].inputSchema["properties"]
    assert "note" in tools["save_note"].inputSchema["properties"]


def test_mcp_save_note_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ehr_api, "get_store_path", lambda: tmp_path / "ehr_store.json")
    monkeypatch.setattr(ehr_client, "get_client", lambda: TestClient(ehr_api.app))

    result = asyncio.run(
        server.mcp.call_tool("save_note", {"patient_id": "P001", "note": "SOAP body"})
    )
    flat = str(result)
    assert "saved" in flat
    assert "N_" in flat

    # The note is really in the store: the history tool sees it.
    history = asyncio.run(
        server.mcp.call_tool("get_patient_history", {"patient_id": "P001"})
    )
    assert "found" in str(history)
