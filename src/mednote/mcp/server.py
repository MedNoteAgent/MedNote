"""MCP server exposing the EHR tools over stdio (Task 13).

Uses the FastMCP API (``mcp`` SDK 1.x) — the low-level ``mcp.server.Server``
class has no ``@tool`` decorator. Both MCP tools delegate to
``mednote.tools.ehr_client``, the same functions behind the LangChain tools,
so the two surfaces can never drift.

Run:  uv run python -m mednote.mcp.server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mednote.tools import ehr_client

mcp = FastMCP("mednote-ehr")


@mcp.tool()
def save_note(patient_id: str, note: str, icd_codes: list[str] | None = None) -> dict:
    """Save a clinical note to the patient's EHR chart.

    Call ONLY after the physician has explicitly asked to save (no auto-save
    without confirmation). Returns the EHR response with the new note ID on
    success.
    """
    return ehr_client.post_note(patient_id, note, icd_codes)


@mcp.tool()
def get_patient_history(patient_id: str) -> dict:
    """Retrieve prior visit notes and history for a patient from the EHR."""
    return ehr_client.fetch_history(patient_id)


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
