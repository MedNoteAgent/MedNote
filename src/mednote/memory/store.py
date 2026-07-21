"""SQLite visit-memory store (Task 14).

One table, ``visits``, keyed by patient — the agent's cross-session memory of
what was documented. Distinct from the mock EHR store (``data/ehr_store.json``):
the EHR is the system of record the *tools* talk to; this is the agent's own
recall, written on every successful save and queried by the ``memory_lookup``
node (Task 15).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mednote.config import get_config


def get_db_path() -> str:
    """Seam for tests to point the store at a temp file."""
    return get_config().memory.db_path


class MemoryStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or get_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_patient ON visits(patient_id)"
            )

    def save_visit(
        self,
        patient_id: str,
        visit_date: str,
        note_id: str | None,
        summary: str,
        icd_codes: list[str],
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO visits (patient_id, visit_date, note_id, summary, icd_codes)"
                " VALUES (?, ?, ?, ?, ?)",
                (patient_id, visit_date, note_id, summary, json.dumps(icd_codes)),
            )
            return cursor.lastrowid

    def get_history(self, patient_id: str) -> list[dict]:
        """Visits for one patient, newest first; ``icd_codes`` parsed back to a list."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM visits WHERE patient_id = ?"
                " ORDER BY visit_date DESC, id DESC",
                (patient_id,),
            ).fetchall()
        return [
            {**dict(row), "icd_codes": json.loads(row["icd_codes"] or "[]")}
            for row in rows
        ]
