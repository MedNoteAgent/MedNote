"""Mock EHR: JSON-file-backed store + FastAPI REST surface (Task 11).

``EHRStore`` is the single implementation of the save/lookup contract in
``docs/tools.md``. The FastAPI app below wraps it as an independently
runnable REST server (``uv run uvicorn mednote.tools.ehr_api:app``); the
LangChain tool wrappers (``tools/save_note.py``, ``tools/get_history.py``)
and the MCP server (``mcp/server.py``) call the store directly in-process
instead of over HTTP — see ``docs/tools.md`` "Implementation notes" for why.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from mednote.config import get_config


class EHRStore:
    """Thread-safe read-modify-write JSON store for saved notes."""

    def __init__(self, store_path: str | None = None):
        self.store_path = Path(store_path or get_config().paths.ehr_store_path)
        self._lock = threading.Lock()

    def _read(self) -> dict:
        if not self.store_path.exists():
            return {"notes": []}
        with self.store_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, data: dict) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def save_note(
        self, patient_id: str, note: str, icd_codes: list[str] | None = None
    ) -> dict:
        """docs/tools.md `save_note` contract."""
        if not patient_id or not patient_id.strip():
            return {"status": "error", "message": "Patient ID is required"}
        if not note or not note.strip():
            return {"status": "error", "message": "Note text is required"}

        record = {
            "note_id": f"N_{uuid.uuid4().hex[:8]}",
            "patient_id": patient_id,
            "note": note,
            "icd_codes": icd_codes or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            store = self._read()
            store.setdefault("notes", []).append(record)
            self._write(store)

        return {
            "status": "saved",
            "note_id": record["note_id"],
            "patient_id": patient_id,
            "timestamp": record["timestamp"],
        }

    def get_history(self, patient_id: str) -> dict:
        """docs/tools.md `get_patient_history` contract."""
        if not patient_id or not patient_id.strip():
            return {"status": "error", "message": "Patient ID is required"}

        with self._lock:
            store = self._read()
        visits = [n for n in store.get("notes", []) if n["patient_id"] == patient_id]
        # Ascending-stable-sort then reverse (not sort(reverse=True), which
        # keeps tied elements in original order): two saves can land on the
        # same microsecond-resolution timestamp, and the more recently
        # inserted one must still come first.
        visits.sort(key=lambda v: v["timestamp"])
        visits.reverse()

        if not visits:
            return {
                "status": "no_history",
                "patient_id": patient_id,
                "message": "No prior visits found",
            }
        return {"status": "found", "patient_id": patient_id, "visits": visits}


@lru_cache(maxsize=1)
def get_ehr_store() -> EHRStore:
    """Process-wide store singleton; tests monkeypatch this factory."""
    return EHRStore()


class SaveNoteRequest(BaseModel):
    patient_id: str
    note: str
    icd_codes: list[str] = []


app = FastAPI(title="MedNote Mock EHR")


@app.post("/notes")
def create_note(payload: SaveNoteRequest):
    result = get_ehr_store().save_note(payload.patient_id, payload.note, payload.icd_codes)
    if result["status"] == "error":
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/patients/{patient_id}/history")
def read_history(patient_id: str):
    return get_ehr_store().get_history(patient_id)
