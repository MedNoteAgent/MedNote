"""FastAPI mock EHR server (Task 11).

Endpoints (response envelopes per docs/implementation_plan.md Task 10 spec):
    POST /notes                       save a note   -> {"status": "saved", "note_id", ...}
    GET  /patients/{id}/history       prior visits  -> {"status": "found" | "no_history", ...}
    GET  /patients/{id}               demographics  -> {"status": "found", "patient": {...}}

The store is a JSON file (config.yml -> paths.ehr_store_path), created on first
write from SEED_PATIENTS — the P001-P013 patients of the Task 4 transcript set
(plus the UI's demo patient), so demographics always line up with the eval
labels. All data is synthetic; no real PHI.

Run:  uv run uvicorn mednote.tools.ehr_api:app --port 8100
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mednote.config import get_config

# Demographics mirror data/transcripts/synthetic_transcripts.json (Task 4);
# P-DEMO mirrors the UI's mock patient chip. Names are synthetic.
SEED_PATIENTS: dict[str, dict] = {
    "P001": {"patient_id": "P001", "name": "Meera Sharma", "age": 34, "sex": "female"},
    "P002": {"patient_id": "P002", "name": "Rajesh Kumar", "age": 58, "sex": "male"},
    "P003": {"patient_id": "P003", "name": "David Okafor", "age": 61, "sex": "male"},
    "P004": {"patient_id": "P004", "name": "Harold Bennett", "age": 65, "sex": "male"},
    "P005": {"patient_id": "P005", "name": "Aarav Patel", "age": 4, "sex": "male"},
    "P006": {"patient_id": "P006", "name": "Lucia Fernandez", "age": 47, "sex": "female"},
    "P007": {"patient_id": "P007", "name": "Susan Miller", "age": 52, "sex": "female"},
    "P008": {"patient_id": "P008", "name": "Priya Nair", "age": 29, "sex": "female"},
    "P009": {"patient_id": "P009", "name": "George Thompson", "age": 68, "sex": "male"},
    "P010": {"patient_id": "P010", "name": "Margaret Wilson", "age": 66, "sex": "female"},
    "P011": {"patient_id": "P011", "name": "Fatima Khan", "age": 45, "sex": "female"},
    "P012": {"patient_id": "P012", "name": "Daniel Reyes", "age": 40, "sex": "male"},
    "P013": {"patient_id": "P013", "name": "Emily Carter", "age": 31, "sex": "female"},
    "P-DEMO": {"patient_id": "P-DEMO", "name": "Sarah Jenkins", "age": 44, "sex": "female"},
}

app = FastAPI(title="MedNote Mock EHR", version="0.1.0")


class NoteIn(BaseModel):
    patient_id: str = Field(min_length=1)
    note: str = Field(min_length=1)
    icd_codes: list[str] = Field(default_factory=list)


def get_store_path() -> Path:
    """Seam for tests to point the store at a temp file."""
    return Path(get_config().paths.ehr_store_path)


def _load_store() -> dict:
    path = get_store_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"patients": dict(SEED_PATIENTS), "notes": []}


def _write_store(store: dict) -> None:
    path = get_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"status": "error", "message": message}
    )


@app.post("/notes")
def save_note(payload: NoteIn):
    store = _load_store()
    if payload.patient_id not in store["patients"]:
        return _error(404, f"Unknown patient ID '{payload.patient_id}'.")

    now = datetime.now(timezone.utc)
    record = {
        "note_id": f"N_{uuid.uuid4().hex[:8]}",
        "patient_id": payload.patient_id,
        "date": now.date().isoformat(),
        "note": payload.note,
        "icd_codes": list(payload.icd_codes),
        "created_at": now.isoformat(),
    }
    _write_store({**store, "notes": [*store["notes"], record]})
    return {
        "status": "saved",
        "note_id": record["note_id"],
        "patient_id": record["patient_id"],
        "timestamp": record["created_at"],
    }


@app.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    patient = _load_store()["patients"].get(patient_id)
    if patient is None:
        return _error(404, f"Unknown patient ID '{patient_id}'.")
    return {"status": "found", "patient": patient}


@app.get("/patients/{patient_id}/history")
def get_history(patient_id: str):
    store = _load_store()
    if patient_id not in store["patients"]:
        return _error(404, f"Unknown patient ID '{patient_id}'.")

    visits = [n for n in store["notes"] if n["patient_id"] == patient_id]
    if not visits:
        return {
            "status": "no_history",
            "patient_id": patient_id,
            "message": "No prior visits found",
        }
    return {"status": "found", "patient_id": patient_id, "visits": visits}
