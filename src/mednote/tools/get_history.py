"""``get_patient_history`` LangChain tool (Task 12). Contract: docs/tools.md."""

from __future__ import annotations

from langchain_core.tools import tool

from mednote.tools.ehr_api import get_ehr_store


@tool
def get_patient_history(patient_id: str) -> dict:
    """Retrieve a patient's prior visit notes from the EHR.

    Returns {"status": "found", "visits": [...]} (newest first) if the
    patient has prior visits, or {"status": "no_history", "message": ...}
    for a new patient.
    """
    return get_ehr_store().get_history(patient_id)
