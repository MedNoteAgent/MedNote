"""SQLite visit-memory store (Task 14).

Schema (one table, ``visits``): each row is one saved encounter for one
patient. This is deliberately separate from the mock EHR's JSON store
(``tools/ehr_api.py``) — the EHR is the system of record a real practice
would have; ``MemoryStore`` is the agent's own recall index, keyed the way
the agent needs to recall it (newest visit per patient, a short summary
ready to drop into a prompt) rather than the way a chart system stores it.
``tool_execution`` (Task 15) writes both on every successful save.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from mednote.config import get_config


class MemoryStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or get_config().memory.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    visit_date TEXT NOT NULL,
                    note_id TEXT,
                    summary TEXT,
                    icd_codes TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_patient ON visits(patient_id)")

    def save_visit(
        self,
        patient_id: str,
        visit_date: str,
        note_id: str | None,
        summary: str,
        icd_codes: list[str] | None = None,
    ) -> int:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO visits (patient_id, visit_date, note_id, summary, icd_codes) "
                "VALUES (?, ?, ?, ?, ?)",
                (patient_id, visit_date, note_id, summary, json.dumps(icd_codes or [])),
            )
            return cursor.lastrowid

    def get_history(self, patient_id: str) -> list[dict]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM visits WHERE patient_id = ? ORDER BY visit_date DESC, id DESC",
                (patient_id,),
            ).fetchall()
            visits = [dict(row) for row in rows]
        for visit in visits:
            visit["icd_codes"] = json.loads(visit["icd_codes"] or "[]")
        return visits
