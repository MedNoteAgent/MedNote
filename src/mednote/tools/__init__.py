"""EHR tool surface (Tasks 10-13): mock EHR API, client, and LangChain tools."""

from mednote.tools.get_history import get_patient_history
from mednote.tools.save_note import save_note

__all__ = ["get_patient_history", "save_note"]
