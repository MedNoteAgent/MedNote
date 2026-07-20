"""Tests for the SQLite visit-memory store (Task 14)."""

from __future__ import annotations

from mednote.memory.store import MemoryStore


def _make_store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "memory.db"))


def test_get_history_empty_for_unknown_patient(tmp_path) -> None:
    store = _make_store(tmp_path)
    assert store.get_history("P001") == []


def test_save_visit_can_be_read_back(tmp_path) -> None:
    store = _make_store(tmp_path)
    visit_id = store.save_visit(
        patient_id="P001",
        visit_date="2026-07-01",
        note_id="N_abc123",
        summary="Headache for 3 days, worse mornings.",
        icd_codes=["G44.1", "R51"],
    )
    assert isinstance(visit_id, int)

    history = store.get_history("P001")
    assert len(history) == 1
    visit = history[0]
    assert visit["patient_id"] == "P001"
    assert visit["visit_date"] == "2026-07-01"
    assert visit["note_id"] == "N_abc123"
    assert visit["summary"] == "Headache for 3 days, worse mornings."
    assert visit["icd_codes"] == ["G44.1", "R51"]


def test_get_history_orders_newest_visit_first(tmp_path) -> None:
    store = _make_store(tmp_path)
    store.save_visit("P001", "2026-01-01", "N_1", "first visit", [])
    store.save_visit("P001", "2026-06-01", "N_2", "second visit", [])

    history = store.get_history("P001")
    assert [v["note_id"] for v in history] == ["N_2", "N_1"]


def test_get_history_scoped_to_patient(tmp_path) -> None:
    store = _make_store(tmp_path)
    store.save_visit("P001", "2026-01-01", "N_1", "for P001", [])
    store.save_visit("P002", "2026-01-01", "N_2", "for P002", [])

    assert [v["note_id"] for v in store.get_history("P001")] == ["N_1"]
    assert [v["note_id"] for v in store.get_history("P002")] == ["N_2"]


def test_store_persists_across_instances(tmp_path) -> None:
    """The SQLite file, not the object, is the source of truth (cross-session recall)."""
    path = str(tmp_path / "memory.db")
    _make_store_at = lambda: MemoryStore(path)  # noqa: E731

    _make_store_at().save_visit("P001", "2026-07-01", "N_1", "session one note", [])
    reopened = _make_store_at()

    history = reopened.get_history("P001")
    assert len(history) == 1
    assert history[0]["summary"] == "session one note"


def test_icd_codes_default_to_empty_list_when_omitted(tmp_path) -> None:
    store = _make_store(tmp_path)
    store.save_visit("P001", "2026-07-01", None, "no codes on this visit")
    assert store.get_history("P001")[0]["icd_codes"] == []
