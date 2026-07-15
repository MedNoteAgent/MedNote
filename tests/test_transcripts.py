"""Tests for the synthetic transcript dataset (Task 4).

The dataset is the project's ground-truth fixture: every entry carries labels
(expected_intent, expected_icd10, is_red_flag, tests) that the eval harness
(Tasks 25-28) scores against. These tests pin the plan's Definition of Done so
a future edit can't silently drop coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mednote.config import get_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "data" / "transcripts" / "synthetic_transcripts.json"
PROCESSED_JSONL = REPO_ROOT / "data" / "icd10_processed" / "icd10_codes.jsonl"

REQUIRED_FIELDS = {
    "transcript_id", "patient_id", "date", "patient_age", "patient_sex",
    "expected_intent", "transcript", "is_red_flag", "expected_icd10",
    "tests", "expected_behavior",
}

VALID_INTENTS = {"soap", "icd_lookup", "save", "history", "refuse"}


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def by_id(entries: list[dict]) -> dict[str, dict]:
    return {e["transcript_id"]: e for e in entries}


def test_dataset_has_all_19_planned_entries(entries: list[dict]) -> None:
    # The plan's tables enumerate TX001-TX019 (19 rows; its prose said "18").
    ids = [e["transcript_id"] for e in entries]
    assert ids == [f"TX{n:03d}" for n in range(1, 20)]


def test_every_entry_carries_the_full_label_schema(entries: list[dict]) -> None:
    for entry in entries:
        missing = REQUIRED_FIELDS - entry.keys()
        assert not missing, f"{entry.get('transcript_id')}: missing {missing}"
        assert entry["expected_intent"] in VALID_INTENTS
        assert entry["patient_sex"] in {"male", "female"}
        assert isinstance(entry["patient_age"], int)
        assert isinstance(entry["is_red_flag"], bool)
        assert isinstance(entry["expected_icd10"], list)
        assert entry["tests"], "every entry needs at least one capability tag"
        assert entry["transcript"].strip()


def test_all_five_graph_intents_are_covered(entries: list[dict]) -> None:
    assert {e["expected_intent"] for e in entries} == VALID_INTENTS


def test_core_six_sample_queries_map_one_to_one(by_id: dict[str, dict]) -> None:
    """requirements.md §3 Q1-Q6 == TX001-TX006."""
    assert by_id["TX001"]["expected_intent"] == "soap"
    assert by_id["TX002"]["expected_intent"] == "icd_lookup"
    assert "G44.2" in by_id["TX002"]["expected_icd10"]  # primary RAG acceptance
    assert by_id["TX003"]["expected_intent"] == "save"
    assert by_id["TX004"]["expected_intent"] == "history"
    assert by_id["TX005"]["is_red_flag"] is True
    assert by_id["TX006"]["expected_intent"] == "refuse"


def test_both_red_flag_families_present(entries: list[dict], by_id: dict[str, dict]) -> None:
    red = {e["transcript_id"] for e in entries if e["is_red_flag"]}
    assert {"TX005", "TX007", "TX016"} <= red     # cardiac x2 + thunderclap/SAH
    assert all(by_id[t]["expected_intent"] == "soap" for t in red), (
        "escalation is a guardrail OUTCOME; red-flag rows enter as intent=soap"
    )


def test_demographic_filter_rows_including_negative_test(by_id: dict[str, dict]) -> None:
    tx13, tx14, tx15 = by_id["TX013"], by_id["TX014"], by_id["TX015"]
    assert tx13["patient_sex"] == "female"
    assert any(c.startswith("O") or c.startswith("Z34") for c in tx13["expected_icd10"])
    assert tx14["patient_sex"] == "male"
    assert any(c.startswith("N40") for c in tx14["expected_icd10"])
    # Negative test: the male-only prostate family must NEVER be expected
    # for the female patient with the same complaint.
    assert tx15["patient_sex"] == "female"
    assert not any(c.startswith("N40") for c in tx15["expected_icd10"])


def test_dosage_guardrail_row_states_no_dose(by_id: dict[str, dict]) -> None:
    tx17 = by_id["TX017"]
    assert "dosage" in " ".join(tx17["tests"]).lower() or "dose" in " ".join(tx17["tests"]).lower()
    # The transcript itself must mention medication WITHOUT any numeric dose.
    assert not any(unit in tx17["transcript"].lower() for unit in ("mg", "milligram", "mcg"))


def test_degradation_rows_bracket_the_word_bounds(by_id: dict[str, dict]) -> None:
    cfg = get_config()
    assert len(by_id["TX018"]["transcript"].split()) < cfg.edge_cases.min_transcript_words
    # TX019 is the noise case: long, with exactly one clinical entity buried inside.
    assert len(by_id["TX019"]["transcript"].split()) > 100


def test_history_scenario_has_a_real_prior_visit(entries: list[dict], by_id: dict[str, dict]) -> None:
    """P001 spans TX001->TX003->TX004 so the history lookup can hit something."""
    p001_visits = [e for e in entries if e["patient_id"] == "P001"]
    assert {e["transcript_id"] for e in p001_visits} >= {"TX001", "TX003", "TX004"}
    dates = {e["transcript_id"]: e["date"] for e in p001_visits}
    assert dates["TX004"] > dates["TX001"], "history visit must postdate the note it recalls"


@pytest.mark.skipif(not PROCESSED_JSONL.exists(), reason="ETL output not present")
def test_every_expected_code_exists_in_the_cms_2026_data(entries: list[dict]) -> None:
    valid = {
        json.loads(line)["code"]
        for line in PROCESSED_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for entry in entries:
        for code in entry["expected_icd10"]:
            assert code in valid, f"{entry['transcript_id']}: {code} not in CMS 2026 tabular"
