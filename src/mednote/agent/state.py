"""LangGraph state schema for the MedNote agent (Task 8).

Design rationale (docs/implementation_plan.md, "State Schema"):
- No ``next_step`` field: routing reads semantic state (``intent``,
  ``guardrail_result``) inside conditional-edge functions, so nodes stay
  decoupled from graph topology.
- Typed payloads (agent/schemas.py), not bare dicts, for the safety-critical
  objects.
- ONE code list: ``suggested_codes`` is the single ranked + specificity-
  expanded result; the raw pre-rerank top-15 goes to the tracer, not state.
- ``errors`` is a reducer channel (operator.add) accumulating soft failures
  (empty transcript, zero-hit, tool error) across nodes.
- ``total=False`` lets every node return a partial dict and lets
  make_initial_state() seed just the required keys.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from mednote.agent.schemas import (
    GuardrailResult,
    MemoryContext,
    SuggestedCode,
    ToolResult,
)

Intent = Literal["soap", "icd_lookup", "save", "history", "refuse"]  # user request only
Sex = Literal["male", "female", "unknown"]


class MedNoteState(TypedDict, total=False):
    # ---- Input (set once at entry) ----
    user_input: str                       # the only always-required key
    intent: Intent
    transcript: str
    patient_id: str

    # Patient demographics (for RAG metadata filtering)
    patient_age: int
    patient_sex: Sex

    # ---- Working (per-request scratch) ----
    extracted_entities: list[str]
    suggested_codes: list[SuggestedCode]  # ONE canonical ranked + expanded list
    draft_note: str
    guardrail_result: GuardrailResult     # incl. is_red_flag (single source of truth)
    tool_result: ToolResult
    memory_context: MemoryContext

    # ---- Output ----
    final_response: str

    # ---- Meta / observability ----
    trace_id: str
    cache_hit: bool
    errors: Annotated[list[str], operator.add]  # reducer channel for soft failures


def make_initial_state(user_input: str, trace_id: str) -> MedNoteState:
    """Seed only the required keys — total=False means no hand-init of Nones."""
    return {"user_input": user_input, "trace_id": trace_id, "errors": []}
