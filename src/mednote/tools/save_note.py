"""``save_note`` LangChain tool (Task 11). Contract: docs/tools.md."""

from __future__ import annotations

from langchain_core.tools import tool

from mednote.tools.ehr_api import get_ehr_store


@tool
def save_note(patient_id: str, note: str, icd_codes: list[str] | None = None) -> dict:
    """Save a clinical note to the patient's EHR chart.

    Only call this after the physician has explicitly confirmed the note is
    ready to save. Returns {"status": "saved", "note_id": ...} on success, or
    {"status": "error", "message": ...} if patient_id or note is missing.
    """
    return get_ehr_store().save_note(patient_id, note, icd_codes or [])
