"""Static content checks for the agent's system prompts.

These don't call an LLM — they verify the prompt text itself follows the
non-diagnostic framing rules from docs/requirements.md §5 (banned phrases
absent, hedging language present, citation/escalation formats correct).
"""

from mednote.agent import prompts


def test_soap_prompt_bans_diagnostic_assertions() -> None:
    body = prompts.SOAP_SYSTEM_PROMPT.lower()
    for phrase in prompts.BANNED_DIAGNOSTIC_PHRASES:
        # Each banned phrase must be named explicitly in the "never write this" rule.
        assert phrase in body


def test_soap_prompt_requires_hedging_language() -> None:
    body = prompts.SOAP_SYSTEM_PROMPT.lower()
    assert any(phrase in body for phrase in prompts.HEDGING_PHRASES)


def test_soap_prompt_requires_icd10_citation_format() -> None:
    assert "(Source:" in prompts.SOAP_SYSTEM_PROMPT
    assert "(Pending Physician Confirmation)" in prompts.SOAP_SYSTEM_PROMPT


def test_soap_prompt_forbids_unstated_dosages() -> None:
    assert "dosage" in prompts.SOAP_SYSTEM_PROMPT.lower()


def test_soap_prompt_never_marks_notes_final() -> None:
    body = prompts.SOAP_SYSTEM_PROMPT.lower()
    assert "save_note" in body
    assert "never mark a note as" in body


def test_icd_lookup_prompt_requires_citation_and_no_guessing() -> None:
    body = prompts.ICD_LOOKUP_SYSTEM_PROMPT
    assert "(Source:" in body
    assert "(Pending Physician Confirmation)" in body
    assert "Do not guess" in body


def test_format_escalation_interpolates_reason() -> None:
    message = prompts.format_escalation("chest pain with radiation to left arm")
    assert "chest pain with radiation to left arm" in message
    assert "URGENT ESCALATION REQUIRED" in message
    assert "emergency evaluation" in message


def test_refusal_prompt_declines_diagnosis_without_asserting_one() -> None:
    body = prompts.REFUSAL_PROMPT.lower()
    assert "cannot provide a definitive diagnosis" in body
    for phrase in prompts.BANNED_DIAGNOSTIC_PHRASES:
        assert phrase not in body
