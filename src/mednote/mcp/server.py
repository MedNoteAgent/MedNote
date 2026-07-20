"""MCP server exposing the EHR tools (Task 13). Contract: docs/tools.md.

Uses ``FastMCP`` (the ``mcp`` SDK's high-level server) rather than the
lower-level ``Server`` + ``stdio_server`` sketch in the implementation plan —
``FastMCP`` derives the tool schema from the function signature and docstring
directly, so there is one definition of each tool's contract instead of a
schema hand-written alongside the implementation.

Run standalone (stdio transport, for any general-purpose MCP client):
    uv run python -m mednote.mcp.server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mednote.tools.ehr_api import get_ehr_store

mcp_server = FastMCP("mednote-ehr")


@mcp_server.tool()
def save_note(patient_id: str, note: str, icd_codes: list[str] | None = None) -> dict:
    """Save a clinical note to the patient's EHR chart.

    Only call this after the physician has explicitly confirmed the note is
    ready to save. Returns {"status": "saved", "note_id": ...} on success, or
    {"status": "error", "message": ...} if patient_id or note is missing.
    """
    return get_ehr_store().save_note(patient_id, note, icd_codes or [])


@mcp_server.tool()
def get_patient_history(patient_id: str) -> dict:
    """Retrieve a patient's prior visit notes from the EHR.

    Returns {"status": "found", "visits": [...]} (newest first) if the
    patient has prior visits, or {"status": "no_history", "message": ...}
    for a new patient.
    """
    return get_ehr_store().get_history(patient_id)


def main() -> None:
    mcp_server.run()


if __name__ == "__main__":
    main()
