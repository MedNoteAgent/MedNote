"""Tests for the prompt library (Task 3).

Prompts are contracts: downstream nodes parse the section headers, the
guardrails depend on the banned/hedging vocabulary, and requirements.md §5
mandates specific behaviors. These tests pin those contracts so a prompt edit
that silently drops a safety rule fails CI.
"""

from __future__ import annotations

import pytest

from mednote.agent.prompts import (
    ESCALATION_PROMPT,
    ICD_LOOKUP_PROMPT,
    REFUSAL_PROMPT,
    SOAP_SYSTEM_PROMPT,
    SOAP_USER_PROMPT,
    format_rag_context,
)

# ------------------------------------------------------- SOAP system prompt ---


def test_soap_prompt_defines_all_five_sections_in_order() -> None:
    headers = [
        "### Subjective",
        "### Objective",
        "### Assessment",
        "### Plan",
        "### Suggested ICD-10 Codes",
    ]
    positions = [SOAP_SYSTEM_PROMPT.find(h) for h in headers]
    assert all(p >= 0 for p in positions), "every SOAP section header must appear"
    assert positions == sorted(positions), "sections must be defined in SOAP order"


def test_soap_prompt_mandates_hedging_and_bans_assertive_phrasing() -> None:
    for hedge in ("may be consistent with", "consider", "possible"):
        assert hedge in SOAP_SYSTEM_PROMPT.lower()
    # The banned-phrase list itself must be spelled out for the model.
    for banned in ('"the patient has"', '"diagnosis is"', '"confirmed"'):
        assert banned in SOAP_SYSTEM_PROMPT


def test_soap_prompt_covers_the_seven_design_principles() -> None:
    """requirements.md §5 obligations, each must be stated explicitly."""
    lower = SOAP_SYSTEM_PROMPT.lower()
    assert "physician review" in lower                    # 2: framing
    assert "pending physician confirmation" in lower      # 5: code marking
    assert "(source:" in lower                            # 5: citation format
    assert "dosage" in lower or "dose" in lower           # 7: no invented dosages
    assert "escalation" in lower                          # 6: red-flag handling
    assert "not documented" in lower                      # missing info handling
    assert "draft" in lower                               # note is a draft


def test_soap_prompt_enumerates_red_flag_criteria() -> None:
    """The model can only flag what it has criteria for (guidelines corpus)."""
    lower = SOAP_SYSTEM_PROMPT.lower()
    for marker in ("chest pain", "thunderclap", "shortness of breath",
                   "neurological", "petechial"):
        assert marker in lower, f"red-flag criterion missing: {marker}"


def test_soap_prompt_forbids_code_fabrication_and_injection() -> None:
    lower = SOAP_SYSTEM_PROMPT.lower()
    assert "only" in lower and "reference" in lower       # codes only from context
    assert "never invent" in lower or "never fabricate" in lower
    # Transcripts are untrusted input; instructions inside them must be ignored.
    assert "not instructions" in lower or "ignore any instructions" in lower


def test_soap_user_prompt_has_exactly_the_expected_slots() -> None:
    rendered = SOAP_USER_PROMPT.format(
        rag_context="RAGCTX", memory_context="MEMCTX", transcript="TRANSCRIPT"
    )
    assert "RAGCTX" in rendered and "MEMCTX" in rendered and "TRANSCRIPT" in rendered
    with pytest.raises(KeyError):
        SOAP_USER_PROMPT.format(transcript="only-one-slot")


# ----------------------------------------------------------- other prompts ---


def test_icd_lookup_prompt_requires_citation_and_confirmation() -> None:
    lower = ICD_LOOKUP_PROMPT.lower()
    assert "cite" in lower or "(source:" in lower
    assert "pending physician confirmation" in lower
    assert "only" in lower and ("provided" in lower or "reference" in lower)


def test_escalation_prompt_formats_reason_and_defers_documentation() -> None:
    rendered = ESCALATION_PROMPT.format(reason="chest pain radiating to left arm")
    assert "chest pain radiating to left arm" in rendered
    assert "URGENT" in rendered
    lower = rendered.lower()
    assert "emergency" in lower
    assert "routine" in lower  # routine documentation is deferred, not skipped


def test_refusal_prompt_declines_but_offers_decision_support() -> None:
    lower = REFUSAL_PROMPT.lower()
    assert "cannot" in lower or "decline" in lower
    assert "differential" in lower
    assert "physician" in lower


# ------------------------------------------------------- RAG context render ---


def _code(code: str = "G44.2", **overrides) -> dict:
    base = {
        "code": code,
        "description": "Tension-type headache",
        "hierarchy_path": "Nervous system -> Episodic disorders",
        "source": "ICD-10-CM 2026",
        "confidence": 0.93,
        "parent_code": "G44",
        "specificity_options": [],
        "pending_confirmation": True,
    }
    return {**base, **overrides}


def test_format_rag_context_renders_citation_and_confirmation() -> None:
    text = format_rag_context([_code()])
    assert "G44.2" in text
    assert "Tension-type headache" in text
    assert "(Source: ICD-10-CM 2026)" in text
    assert "(Pending Physician Confirmation)" in text
    assert "confidence 0.93" in text


def test_format_rag_context_renders_specificity_options() -> None:
    text = format_rag_context(
        [
            _code(
                specificity_options=[
                    _code("G44.21", description="Episodic tension-type headache"),
                    _code("G44.22", description="Chronic tension-type headache"),
                ]
            )
        ]
    )
    assert "G44.21" in text and "G44.22" in text
    assert "more specific" in text.lower()


def test_format_rag_context_zero_hit_says_assign_manually() -> None:
    text = format_rag_context([])
    assert "manually" in text.lower()
    assert "insufficient" in text.lower()
