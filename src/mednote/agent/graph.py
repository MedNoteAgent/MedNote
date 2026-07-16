"""StateGraph assembly for the MedNote agent (Task 8).

MVP flow (docs/implementation_plan.md):

    user_input -> parse_input ──(intent router)──┬─ soap ──────► context_extraction
                                                 ├─ icd_lookup ► entity_extraction
                                                 ├─ save ──────► tool_execution ────────┐
                                                 ├─ history ───► memory_lookup ─────────┤
                                                 └─ refuse ────► response_generation    │
    context_extraction ► entity_extraction ► rag_pipeline ──(intent)──┬─ soap ► note_generation ► guardrail_check ─┤
                                                                      └─ icd_lookup ───────────────────────────────┤
                                                                                       response_generation ► END ◄─┘

Routers read semantic state (``intent``) only — no next_step field, so nodes
stay decoupled from topology.
"""

from __future__ import annotations

import uuid
from functools import lru_cache

from langgraph.graph import END, StateGraph

from mednote.agent.nodes import (
    context_extraction,
    entity_extraction,
    guardrail_check,
    memory_lookup,
    note_generation,
    parse_input,
    rag_pipeline,
    response_generation,
    tool_execution,
)
from mednote.agent.state import MedNoteState, make_initial_state


def build_graph():
    graph = StateGraph(MedNoteState)

    graph.add_node("parse_input", parse_input)
    graph.add_node("context_extraction", context_extraction)
    graph.add_node("entity_extraction", entity_extraction)
    graph.add_node("rag_pipeline", rag_pipeline)
    graph.add_node("note_generation", note_generation)
    graph.add_node("guardrail_check", guardrail_check)
    graph.add_node("tool_execution", tool_execution)
    graph.add_node("memory_lookup", memory_lookup)
    graph.add_node("response_generation", response_generation)

    graph.set_entry_point("parse_input")
    graph.add_conditional_edges(
        "parse_input",
        lambda s: s["intent"],
        {
            "soap": "context_extraction",
            "icd_lookup": "entity_extraction",
            "save": "tool_execution",
            "history": "memory_lookup",
            "refuse": "response_generation",
        },
    )
    graph.add_edge("context_extraction", "entity_extraction")
    graph.add_edge("entity_extraction", "rag_pipeline")
    graph.add_conditional_edges(
        "rag_pipeline",
        lambda s: s["intent"],
        {
            "soap": "note_generation",
            "icd_lookup": "response_generation",
        },
    )
    graph.add_edge("note_generation", "guardrail_check")
    # response_generation reads guardrail_result to format routine vs escalation.
    graph.add_edge("guardrail_check", "response_generation")
    graph.add_edge("tool_execution", "response_generation")
    graph.add_edge("memory_lookup", "response_generation")
    graph.add_edge("response_generation", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_compiled_graph():
    """Compile once per process; the graph itself is stateless."""
    return build_graph()


def run_agent(
    user_input: str,
    patient_id: str | None = None,
    patient_age: int | None = None,
    patient_sex: str | None = None,
    trace_id: str | None = None,
) -> MedNoteState:
    """One full turn: seed state, invoke the graph, return the final state.

    Demographics are optional; until the mock EHR lands (Task 11) the caller
    (UI / eval harness) supplies them from its own records.
    """
    state = make_initial_state(user_input, trace_id or str(uuid.uuid4()))
    if patient_id is not None:
        state["patient_id"] = patient_id
    if patient_age is not None:
        state["patient_age"] = patient_age
    if patient_sex is not None:
        state["patient_sex"] = patient_sex
    return get_compiled_graph().invoke(state)
