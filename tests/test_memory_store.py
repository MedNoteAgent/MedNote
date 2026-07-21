"""Tests for the SQLite visit-memory store (Task 14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mednote.memory.store import MemoryStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(db_path=str(tmp_path / "memory.db"))


def test_save_and_get_roundtrip_parses_codes(store: MemoryStore) -> None:
    row_id = store.save_visit(
        patient_id="P001",
        visit_date="2026-07-19",
        note_id="N_abc123",
        summary="Possible tension-type headache — physician review.",
        icd_codes=["G44.2", "R51"],
    )
    assert row_id == 1

    history = store.get_history("P001")
    assert len(history) == 1
    visit = history[0]
    assert visit["note_id"] == "N_abc123"
    assert visit["icd_codes"] == ["G44.2", "R51"]  # list, not a JSON string
    assert visit["summary"].startswith("Possible tension-type")


def test_history_is_newest_first(store: MemoryStore) -> None:
    store.save_visit("P001", "2026-07-01", "N_old", "older visit", [])
    store.save_visit("P001", "2026-07-19", "N_new", "newer visit", [])
    dates = [v["visit_date"] for v in store.get_history("P001")]
    assert dates == ["2026-07-19", "2026-07-01"]


def test_history_is_per_patient(store: MemoryStore) -> None:
    store.save_visit("P001", "2026-07-19", "N_1", "P001 visit", [])
    store.save_visit("P002", "2026-07-19", "N_2", "P002 visit", [])
    assert [v["note_id"] for v in store.get_history("P001")] == ["N_1"]
    assert store.get_history("P999") == []


def test_persists_across_instances(tmp_path: Path) -> None:
    db = str(tmp_path / "memory.db")
    MemoryStore(db_path=db).save_visit("P001", "2026-07-19", "N_1", "visit", [])
    assert len(MemoryStore(db_path=db).get_history("P001")) == 1
