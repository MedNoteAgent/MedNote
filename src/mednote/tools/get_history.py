"""LangChain tool: retrieve patient visit history from the mock EHR (Task 12)."""

from __future__ import annotations

from langchain_core.tools import tool

from mednote.tools import ehr_client


@tool
def get_patient_history(patient_id: str) -> dict:
    """Retrieve prior visit notes and history for a patient from the EHR.

    Returns the EHR response: prior visits when found, or a no-history
    message for patients without saved notes.
    """
    return ehr_client.fetch_history(patient_id)
