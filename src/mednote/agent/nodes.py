"""LangGraph node functions for the MedNote agent (Task 8).

Every node takes the state and returns a PARTIAL update dict (total=False
state). Nodes never name downstream nodes — routing lives in graph.py and
reads semantic state (intent, guardrail_result).

Heavy services (SapBERT, Qdrant, the LLMs) are built lazily by the cached
``get_*`` factories below; tests monkeypatch the factories with fakes.

Stubs, replaced by later tasks:
    guardrail_check   Task 18 (deterministic red-flag + dosage rules)
    memory_lookup     Tasks 14-15 (visit memory)
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

from mednote.agent.prompts import (
    ESCALATION_PROMPT,
    REFUSAL_PROMPT,
    SOAP_SYSTEM_PROMPT,
    SOAP_USER_PROMPT,
    TOOL_SYSTEM_PROMPT,
    format_rag_context,
)
from mednote.agent.state import MedNoteState
from mednote.rag.pipeline import ZERO_HIT_MESSAGE

logger = logging.getLogger(__name__)

INSUFFICIENT_INPUT_MESSAGE = (
    "Insufficient transcript content to generate a reliable note. "
    "Please provide the full encounter transcript."
)

_SAVE_KEYWORDS = ("save", "chart", "store")
_ICD_KEYWORDS = ("icd", "code", "coding")
_HISTORY_KEYWORDS = ("history", "last visit", "prior", "previous")
_REFUSE_KEYWORDS = ("diagnose", "diagnosis", "what does the patient have")
_DIALOGUE_MARKERS = ("doctor:", "patient:", "parent:")
_TRANSCRIPT_WORD_HINT = 40


# ------------------------------------------------------- service factories ---


_rag_pipeline = None
_rag_pipeline_lock = threading.Lock()


def get_rag_pipeline():
    """Real RAG stack (SapBERT + BM25 + embedded Qdrant); built once.

    Double-checked locking, NOT lru_cache: concurrent first calls (the UI's
    warm-up thread racing a "Generate Note" click) would both execute an
    lru_cache-wrapped body, and the second embedded QdrantClient on the same
    path fails with portalocker.AlreadyLocked even within one process.
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        with _rag_pipeline_lock:
            if _rag_pipeline is None:
                _rag_pipeline = _build_rag_pipeline()
    return _rag_pipeline


def close_rag_pipeline() -> bool:
    """Close the embedded Qdrant client and drop the cached pipeline.

    Releases the single-process lock on data/qdrant_data/ so CLI scripts or
    another notebook kernel can open the store. Returns True if a pipeline
    was actually closed; safe to call when nothing was ever built.
    """
    global _rag_pipeline
    with _rag_pipeline_lock:
        if _rag_pipeline is None:
            return False
        try:
            _rag_pipeline.retriever.client.close()
        finally:
            _rag_pipeline = None
        return True


def _build_rag_pipeline():
    from mednote.rag.cache import RAGCache
    from mednote.rag.embeddings import Bm25SparseEncoder, ClinicalEmbedder
    from mednote.rag.entity_extractor import EntityExtractor
    from mednote.rag.indexer import get_qdrant_client
    from mednote.rag.pipeline import RAGPipeline
    from mednote.rag.reranker import ClinicalReranker
    from mednote.rag.retriever import HybridRetriever
    from mednote.rag.specificity import SpecificityChecker

    client = get_qdrant_client()
    embedder = ClinicalEmbedder()
    return RAGPipeline(
        entity_extractor=EntityExtractor(),
        retriever=HybridRetriever(client, embedder, Bm25SparseEncoder()),
        reranker=ClinicalReranker(),
        specificity_checker=SpecificityChecker(client),
        cache=RAGCache(),
    )


@lru_cache(maxsize=1)
def get_note_llm():
    """Main LLM for SOAP generation (config.yml -> llm)."""
    from mednote.llm.wrapper import get_llm

    return get_llm()


@lru_cache(maxsize=1)
def get_tool_llm():
    """Main LLM bound to the EHR tools (Tasks 11-13)."""
    from mednote.llm.wrapper import get_llm
    from mednote.tools import get_patient_history, save_note

    return get_llm().bind_tools([save_note, get_patient_history])


def _llm_text(message) -> str:
    """Flatten str-or-content-blocks message content (Gemini returns blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in content
    )


# ------------------------------------------------------------------- nodes ---


def parse_input(state: MedNoteState) -> dict:
    """Intent classification. Writes intent only; the router maps it onward.

    Deviation from the plan's keyword sketch: a pasted dialogue transcript is
    classified as ``soap`` BEFORE keyword matching — clinical dialogue
    routinely contains trigger words ("any HISTORY of heart problems?",
    "we'll CODE this later") that would misroute an entire encounter.
    """
    text = state["user_input"]
    lower = text.lower()

    is_dialogue = any(marker in lower for marker in _DIALOGUE_MARKERS)
    if is_dialogue or len(text.split()) > _TRANSCRIPT_WORD_HINT:
        return {"intent": "soap", "transcript": text}
    if any(kw in lower for kw in _SAVE_KEYWORDS):
        return {"intent": "save"}
    if any(kw in lower for kw in _ICD_KEYWORDS):
        return {"intent": "icd_lookup", "transcript": text}
    if any(kw in lower for kw in _HISTORY_KEYWORDS):
        return {"intent": "history"}
    if any(kw in lower for kw in _REFUSE_KEYWORDS):
        return {"intent": "refuse"}
    return {"intent": "soap", "transcript": text}


def context_extraction(state: MedNoteState) -> dict:
    """Patient demographics for RAG hard-filtering (Step 7.1).

    Caller-supplied demographics (UI / eval harness) win; otherwise the mock
    EHR (Task 11) is asked by patient ID. Anything still unknown stays
    unknown — the retriever never excludes on missing information, and an
    unreachable EHR degrades gracefully instead of blocking the note.
    """
    if state.get("patient_sex"):
        return {}

    patient_id = state.get("patient_id")
    if patient_id:
        from mednote.tools.ehr_client import EhrApiError, fetch_demographics

        try:
            result = fetch_demographics(patient_id)
        except EhrApiError as exc:
            logger.warning("EHR demographics unavailable (%s); proceeding unfiltered", exc)
        else:
            if result.get("status") == "found":
                patient = result["patient"]
                updates: dict = {"patient_sex": patient.get("sex", "unknown")}
                if state.get("patient_age") is None and patient.get("age") is not None:
                    updates["patient_age"] = patient["age"]
                return updates
    return {"patient_sex": "unknown"}


def entity_extraction(state: MedNoteState) -> dict:
    """Normalize the transcript into formal clinical entities (Step 7.2)."""
    from mednote.config import get_config

    transcript = (state.get("transcript") or "").strip()
    if (
        state.get("intent") == "soap"
        and len(transcript.split()) < get_config().edge_cases.min_transcript_words
    ):
        return {"extracted_entities": [], "errors": [INSUFFICIENT_INPUT_MESSAGE]}

    extractor = get_rag_pipeline().entity_extractor
    try:
        entities = extractor.extract(transcript)
    except ValueError as exc:
        logger.warning("Entity extraction failed (%s); using raw transcript", exc)
        entities = [transcript]
    return {"extracted_entities": entities or [transcript]}


def rag_pipeline(state: MedNoteState) -> dict:
    """Retrieve + rerank + specificity-expand into suggested_codes."""
    entities = state.get("extracted_entities") or []
    if not entities:
        return {"suggested_codes": [], "cache_hit": False}

    pipeline = get_rag_pipeline()
    hits_before = pipeline.cache.hits
    codes = pipeline.run(
        state["transcript"],
        patient_sex=state.get("patient_sex", "unknown"),
        patient_age=state.get("patient_age"),
        entities=entities,
    )
    updates: dict = {
        "suggested_codes": codes,
        "cache_hit": pipeline.cache.hits > hits_before,
    }
    if not codes:
        updates["errors"] = [ZERO_HIT_MESSAGE]
    return updates


def note_generation(state: MedNoteState) -> dict:
    """Draft the SOAP note with RAG results injected (Task 3 prompts)."""
    if INSUFFICIENT_INPUT_MESSAGE in (state.get("errors") or []):
        # Below the input floor: degrade gracefully, never invent a note.
        return {"draft_note": INSUFFICIENT_INPUT_MESSAGE}

    response = get_note_llm().invoke(
        [
            ("system", SOAP_SYSTEM_PROMPT),
            (
                "human",
                SOAP_USER_PROMPT.format(
                    rag_context=format_rag_context(state.get("suggested_codes") or []),
                    transcript=state["transcript"],
                ),
            ),
        ]
    )
    return {"draft_note": _llm_text(response)}


def guardrail_check(_state: MedNoteState) -> dict:
    """STUB until Task 18: passes everything through as clean.

    The SOAP system prompt independently instructs escalation-first output
    for red flags, so the demo still escalates — but the deterministic,
    authoritative check lands with Task 18.
    """
    return {
        "guardrail_result": {
            "passed": True,
            "is_red_flag": False,
            "severity": "info",
            "flags": [],
        }
    }


def tool_execution(state: MedNoteState) -> dict:
    """Execute the physician's explicit EHR request via LLM tool calling (Task 13).

    Reached only on ``intent="save"`` — the routing itself is the guardrail-G5
    confirmation gate (no auto-save: a save happens only because the physician
    asked for one). The bound LLM picks the tool + arguments from the request
    context; we execute it and report a typed ToolResult. An unreachable EHR
    degrades to a clear "NOT saved" message on the errors channel.
    """
    from mednote.tools import get_patient_history, save_note
    from mednote.tools.ehr_client import EhrApiError

    registry = {"save_note": save_note, "get_patient_history": get_patient_history}

    context_lines = [f"Request: {state['user_input']}"]
    if state.get("patient_id"):
        context_lines.append(f"Patient ID: {state['patient_id']}")
    if state.get("draft_note"):
        context_lines.append(f"Draft note to save:\n{state['draft_note']}")
    codes = [c["code"] for c in state.get("suggested_codes") or []]
    if codes:
        context_lines.append(f"Suggested ICD-10 codes: {', '.join(codes)}")

    response = get_tool_llm().invoke(
        [("system", TOOL_SYSTEM_PROMPT), ("human", "\n\n".join(context_lines))]
    )

    tool_calls = getattr(response, "tool_calls", None) or []
    if not tool_calls:
        detail = _llm_text(response).strip() or (
            "No EHR action was taken — the request did not contain enough "
            "information (a patient ID and note content are required)."
        )
        return {
            "tool_result": {"ok": False, "detail": detail, "note_id": None},
            "errors": [detail],
        }

    call = tool_calls[0]
    tool = registry.get(call["name"])
    if tool is None:
        detail = f"Unknown tool requested: '{call['name']}'. No EHR action was taken."
        return {
            "tool_result": {"ok": False, "detail": detail, "note_id": None},
            "errors": [detail],
        }

    try:
        result = tool.invoke(call["args"])
    except EhrApiError as exc:
        detail = (
            "Unable to reach the EHR at this time — the note was NOT saved. "
            f"Please try again or save manually. ({exc})"
        )
        return {
            "tool_result": {"ok": False, "detail": detail, "note_id": None},
            "errors": [detail],
        }

    return _format_tool_result(call["name"], result)


def _format_tool_result(tool_name: str, result: dict) -> dict:
    """Map an EHR API envelope onto a typed ToolResult update."""
    status = result.get("status")

    if tool_name == "save_note" and status == "saved":
        detail = (
            f"Note saved to patient {result['patient_id']}'s chart "
            f"(note ID {result['note_id']})."
        )
        return {"tool_result": {"ok": True, "detail": detail, "note_id": result["note_id"]}}

    if tool_name == "get_patient_history":
        if status == "found":
            lines = [f"Prior visits for patient {result['patient_id']}:"]
            for visit in result.get("visits", []):
                visit_codes = ", ".join(visit.get("icd_codes") or []) or "no codes"
                lines.append(
                    f"- {visit.get('date')}: note {visit.get('note_id')} ({visit_codes})"
                )
            return {
                "tool_result": {"ok": True, "detail": "\n".join(lines), "note_id": None}
            }
        if status == "no_history":
            detail = result.get("message", "No prior visits found.")
            return {"tool_result": {"ok": True, "detail": detail, "note_id": None}}

    detail = result.get("message") or f"EHR request failed: {result}"
    return {
        "tool_result": {"ok": False, "detail": detail, "note_id": None},
        "errors": [detail],
    }


def memory_lookup(state: MedNoteState) -> dict:
    """STUB until Tasks 14-15 (visit memory)."""
    return {
        "memory_context": {
            "patient_id": state.get("patient_id", ""),
            "prior_visits": [],
            "summary": (
                "No prior visit records are available yet — visit memory "
                "arrives with Tasks 14-15."
            ),
        }
    }


def response_generation(state: MedNoteState) -> dict:
    """Format the final reply from semantic state; last node before END."""
    intent = state.get("intent")

    if intent == "refuse":
        return {"final_response": REFUSAL_PROMPT}

    if intent == "save":
        tool_result = state.get("tool_result") or {}
        return {"final_response": tool_result.get("detail", "No tool result available.")}

    if intent == "history":
        memory = state.get("memory_context") or {}
        return {"final_response": memory.get("summary", "No history available.")}

    if intent == "icd_lookup":
        codes = state.get("suggested_codes") or []
        if not codes:
            return {"final_response": ZERO_HIT_MESSAGE}
        return {
            "final_response": "Suggested ICD-10 codes:\n" + format_rag_context(codes)
        }

    # soap (default): the draft note, prefixed by an escalation banner when
    # the guardrail flagged the encounter (stub never does until Task 18).
    note = state.get("draft_note") or INSUFFICIENT_INPUT_MESSAGE
    guardrail = state.get("guardrail_result")
    if guardrail and guardrail.get("is_red_flag"):
        reason = "; ".join(guardrail.get("flags") or ["red-flag symptoms detected"])
        note = ESCALATION_PROMPT.format(reason=reason) + "\n\n" + note
    return {"final_response": note}
