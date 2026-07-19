"""LangChain tool: save a clinical note to the mock EHR (Task 11)."""

from __future__ import annotations

from langchain_core.tools import tool

from mednote.tools import ehr_client


@tool
def save_note(patient_id: str, note: str, icd_codes: list[str] | None = None) -> dict:
    """Save a clinical note to the patient's EHR chart.

    Call ONLY after the physician has explicitly asked to save (no auto-save
    without confirmation). Returns the EHR response; on success it contains
    the new note ID.
    """
    return ehr_client.post_note(patient_id, note, icd_codes)
