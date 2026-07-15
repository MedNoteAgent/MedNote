"""Typed state payloads shared by the RAG pipeline and the LangGraph nodes.

TypedDicts, not dataclasses: plain dicts at runtime (no checkpoint /
serialization friction) but statically type-checked, so a key typo like
``hierarchy`` vs ``hierarchy_path`` fails at author time instead of becoming
a runtime KeyError. Field names deliberately mirror the ETL ``ICD10Code``
dataclass (code, description, hierarchy_path, parent_code, children_codes)
so no naming drift is possible between the ETL output and what nodes read.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class SuggestedCode(TypedDict, total=False):
    code: str
    description: str
    hierarchy_path: str          # matches ICD10Code — single spelling across ETL + nodes
    source: str                  # e.g. "ICD-10-CM 2026" (citation, per §5 guardrail 4)
    confidence: float            # per-code re-rank confidence
    parent_code: str | None
    specificity_options: list["SuggestedCode"]  # laterality children (Step 7.5)
    pending_confirmation: bool   # True until physician sign-off


class GuardrailResult(TypedDict):
    passed: bool
    is_red_flag: bool            # SINGLE source of truth (no duplicate top-level field)
    severity: Literal["info", "warning", "error"]  # info=clean, warning=reframe, error=block
    flags: list[str]


class ToolResult(TypedDict):
    ok: bool                     # typed success/failure, not a bare string
    detail: str
    note_id: str | None          # populated by save_note


class MemoryContext(TypedDict):
    patient_id: str
    prior_visits: list[dict]
    summary: str
