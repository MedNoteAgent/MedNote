"""Gradio UI for MedNote Scribe (tasks.md Task 9).

"Generate Note" runs the full Task 8 LangGraph agent (intent routing -> RAG
over the ICD-10 index -> SOAP generation -> guardrail stub) via run_agent().
"Start Dictation" remains a stub since live audio transcription is out of
scope for the demo (requirements.md §4).

Launch (from the repo root, after `scripts/build_index.py` has been run):
    uv run python -m mednote.ui.app
"""

import html
import logging
import re
import threading

import gradio as gr

from mednote.agent.graph import run_agent

logger = logging.getLogger(__name__)

# Demo patient shown in the header chip; demographics feed the RAG demographic
# filter until the mock EHR lands (Task 11).
MOCK_PATIENT_ID = "P-DEMO"
MOCK_PATIENT_NAME = "Sarah Jenkins"
MOCK_PATIENT_DOB = "05/12/1982"
MOCK_PATIENT_AGE = 44
MOCK_PATIENT_SEX = "female"

ROUTINE_TRANSCRIPT = """Doctor: What brings you in today?
Patient: I've had a headache for about 3 days now. It's worse in the morning and gets a bit better by afternoon.
Doctor: Any nausea, vision changes, or sensitivity to light?
Patient: No nausea, no vision problems. Maybe a little sensitive to light.
Doctor: Let's check your vitals. Blood pressure is 130 over 85. Pulse is normal.
Patient: Is that okay?
Doctor: It's slightly elevated but not concerning on its own. Let's talk about the headache pattern some more."""

EMERGENCY_TRANSCRIPT = """Doctor: What's going on today?
Patient: I've had this chest pain for the last twenty minutes. It's radiating down my left arm.
Doctor: Any shortness of breath, sweating, or nausea with it?
Patient: Yes, I've been sweating and I feel a bit nauseous.
Doctor: Okay, I want to act on this right away. When exactly did the pain start?
Patient: About twenty minutes ago, out of nowhere, while I was sitting down."""

CSS = """
.gradio-container {
    background: #f3f5f8 !important;
    max-width: 100% !important;
    color-scheme: light !important;
}
#transcript-box textarea {
    background: white !important;
    color: #1a2233 !important;
}
#app-header {
    background: white;
    border-radius: 14px;
    border: 1px solid #e5e9ef;
    padding: 18px 24px !important;
    margin-bottom: 18px;
    align-items: center !important;
}
#app-title-block {
    display: flex;
    align-items: center;
    gap: 14px;
}
#app-logo {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #4f7bf3, #3358d6);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
#app-titles h1 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 700;
    color: #1a2233;
    line-height: 1.2;
}
#app-titles span {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #8a93a6;
}
#patient-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #f3f5f8;
    border: 1px solid #e5e9ef;
    border-radius: 999px;
    padding: 8px 18px;
    justify-content: flex-end;
    margin-left: auto;
    white-space: nowrap;
}
#patient-chip .name {
    font-weight: 600;
    color: #1a2233;
    font-size: 0.9rem;
}
#patient-chip .dob {
    color: #8a93a6;
    font-size: 0.85rem;
    border-left: 1px solid #d7dce4;
    padding-left: 10px;
}
.panel-card {
    background: white;
    border-radius: 14px;
    border: 1px solid #e5e9ef;
    padding: 0 !important;
    overflow: hidden;
}
.panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid #eef1f5;
}
.panel-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: #1a2233;
    display: flex;
    align-items: center;
    gap: 8px;
}
#transcript-box textarea {
    border: none !important;
    box-shadow: none !important;
    min-height: 380px !important;
    font-size: 0.95rem;
    padding: 20px !important;
}
#quick-fill-row {
    padding: 12px 20px 0 20px !important;
    gap: 8px !important;
}
#quick-fill-row button {
    border-radius: 999px !important;
    font-size: 0.8rem !important;
    padding: 6px 14px !important;
    background: white !important;
    border: 1px solid #d7dce4 !important;
    color: #3a4356 !important;
    box-shadow: none !important;
}
#action-row {
    padding: 14px 20px !important;
    border-top: 1px solid #eef1f5;
    gap: 12px !important;
}
#dictation-btn {
    background: white !important;
    border: 1px solid #cdd8fb !important;
    color: #3358d6 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
#generate-btn {
    background: #26304a !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
#note-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 80px 40px;
    color: #8a93a6;
    min-height: 380px;
}
#note-empty-state h3 {
    color: #4a5468 !important;
    font-size: 1.1rem;
    margin: 16px 0 8px 0;
}
#note-empty-state p {
    max-width: 340px;
    font-size: 0.9rem;
    line-height: 1.5;
    margin: 0;
    color: #8a93a6 !important;
}
#emergency-output {
    padding: 20px 20px 0 20px !important;
}
.emergency-banner {
    background: #fdeceb;
    border: 1px solid #f3c0bc;
    border-left: 5px solid #d93025;
    border-radius: 10px;
    padding: 16px 18px;
}
.emergency-banner .emergency-title {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #b3261e;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 0.02em;
    margin-bottom: 8px;
}
.emergency-banner .emergency-body {
    color: #b3261e;
    font-weight: 600;
    font-size: 0.95rem;
    line-height: 1.65;
}
.emergency-banner .emergency-body strong {
    color: #d93025;
    font-weight: 800;
    text-decoration: underline;
    text-underline-offset: 3px;
}
#note-output {
    padding: 20px !important;
    color: #2a3346 !important;
}
#note-output h1, #note-output h2, #note-output h3, #note-output h4 {
    color: #1a2233 !important;
}
#note-output p, #note-output li {
    color: #2a3346 !important;
}
#note-output blockquote {
    color: #5a6478 !important;
    border-left: 3px solid #d7dce4 !important;
    padding-left: 12px !important;
    margin-left: 0 !important;
}
#note-output em, #note-output strong {
    color: inherit !important;
}
#codes-output {
    padding: 0 20px 20px 20px !important;
}
#codes-panel {
    border-top: 1px solid #eef1f5;
    padding-top: 16px;
}
#codes-panel h3 {
    margin: 0 0 10px 0;
    font-size: 1rem;
    font-weight: 600;
    color: #1a2233;
}
#codes-panel .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.code-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #eef2fb;
    border: 1px solid #cdd8fb;
    color: #26304a;
    border-radius: 999px;
    padding: 7px 16px;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: help;
}
.code-chip input[type="checkbox"] {
    accent-color: #3358d6;
    width: 15px;
    height: 15px;
    margin: 0;
    cursor: pointer;
}
.codes-footer {
    color: #8a93a6 !important;
    font-size: 0.8rem;
    line-height: 1.6;
    margin: 12px 0 0 0;
}
"""

PULSE_ICON = """<svg width="22" height="22" viewBox="0 0 24 24" fill="none"
    stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 12h4l2 8 4-16 2 8h6" />
</svg>"""

DOC_ICON_LARGE = """<svg width="52" height="52" viewBox="0 0 24 24" fill="none"
    stroke="#c4cad6" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <path d="M14 2v6h6"/>
</svg>"""

PERSON_ICON = """<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
    stroke="#4a5468" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <circle cx="12" cy="8" r="4"/>
    <path d="M4 21v-1a8 8 0 0 1 16 0v1"/>
</svg>"""

PULSE_ICON_SMALL = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none"
    stroke="#3a4356" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 12h4l2 8 4-16 2 8h6" />
</svg>"""


def load_routine():
    return ROUTINE_TRANSCRIPT


def load_emergency():
    return EMERGENCY_TRANSCRIPT


def start_dictation():
    gr.Info("Live dictation isn't available in this demo — paste or type the transcript instead.")


def strip_codes_section(note: str) -> str:
    """Remove the LLM's inline '### Suggested ICD-10 Codes' section.

    The UI renders codes from the STRUCTURED suggested_codes state instead —
    chips with hover descriptions — so the prose version would be duplication.
    """
    marker = "### Suggested ICD-10 Codes"
    start = note.find(marker)
    if start == -1:
        return note
    rest = note[start + len(marker):]
    next_section = rest.find("### ")
    if next_section == -1:
        return note[:start].rstrip()
    return (note[:start] + rest[next_section:]).rstrip()


# Phrases that mark an escalation preamble: the SOAP prompt's rule-4 output
# and the guardrail's ESCALATION_PROMPT both lead with "URGENT ESCALATION".
_EMERGENCY_MARKERS = ("urgent escalation", "red-flag", "red flag", "🚨")


def split_escalation(note: str) -> tuple[str | None, str]:
    """Split a leading escalation warning off the note.

    Emergency encounters arrive with the warning ABOVE the SOAP sections
    (SOAP prompt rule 4, or the guardrail's ESCALATION_PROMPT prefix once
    Task 18 lands). Returns (escalation_text, remaining_note); escalation_text
    is None for routine notes.
    """
    idx = note.find("### Subjective")
    if idx == -1:
        idx = note.find("### ")
    preamble, remainder = (note, "") if idx == -1 else (note[:idx], note[idx:])
    preamble = preamble.strip()
    if preamble and any(marker in preamble.lower() for marker in _EMERGENCY_MARKERS):
        return preamble, remainder.strip()
    return None, note


def render_emergency_banner(escalation_text: str) -> str:
    """Render the escalation warning as a red alert banner above the note.

    The named risks (the model bolds them with **...**) become <strong>
    elements inside the banner, so they read bold-on-red at a glance.
    """
    body = html.escape(escalation_text)
    body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", body)
    body = re.sub(r"^#+\s*", "", body, flags=re.MULTILINE)  # headings would leak as '#'
    # Drop a leading line that just repeats the banner title.
    lines = body.splitlines()
    if lines and "urgent escalation" in lines[0].lower():
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    body = "<br>".join(lines).strip()
    return (
        '<div class="emergency-banner" role="alert">'
        '<div class="emergency-title">🚨 EMERGENCY — URGENT ESCALATION REQUIRED</div>'
        f'<div class="emergency-body">{body}</div>'
        "</div>"
    )


def render_code_chips(codes: list[dict]) -> str:
    """Render suggested codes as hoverable chips with confirm checkboxes.

    Chip text is the bare code; the description (+ confidence and any
    more-specific children) lives in the hover tooltip. The source is cited
    once, in the footer, instead of on every line.
    """
    if not codes:
        return ""

    chips = []
    for code in codes:
        tooltip = f"{code.get('description', '')} — confidence {code.get('confidence', 0.0):.2f}"
        options = code.get("specificity_options") or []
        if options:
            tooltip += " | More specific: " + ", ".join(
                f"{o['code']} ({o.get('description', '')})" for o in options
            )
        chips.append(
            f'<label class="code-chip" title="{html.escape(tooltip, quote=True)}">'
            f'<input type="checkbox" aria-label="Physician confirms {html.escape(code["code"])}">'
            f"{html.escape(code['code'])}</label>"
        )

    source = html.escape(codes[0].get("source", "ICD-10-CM 2026"))
    return (
        '<div id="codes-panel">'
        "<h3>Suggested ICD-10 Codes</h3>"
        f'<div class="chips">{"".join(chips)}</div>'
        '<p class="codes-footer">Hover a code for its description. '
        "All codes are <em>Pending Physician Confirmation</em> — tick a box to confirm.<br>"
        f"Source: {source}</p>"
        "</div>"
    )


def build_trace(final: dict) -> dict:
    """Task 16: what the graph retrieved/called/recalled, for the trace panel.

    Node-by-node timing is Week 4's job (Task 24 observability); this is the
    Week 2 slice — which RAG codes were retrieved, which tool was called with
    what result, and what memory was used to build the response.
    """
    codes = final.get("suggested_codes") or []
    memory = final.get("memory_context") or {}
    return {
        "intent": final.get("intent"),
        "extracted_entities": final.get("extracted_entities", []),
        "rag_retrieval": [
            {
                "code": c.get("code"),
                "description": c.get("description"),
                "confidence": c.get("confidence"),
                "source": c.get("source"),
            }
            for c in codes
        ],
        "cache_hit": final.get("cache_hit", False),
        "memory_used": {
            "patient_id": memory.get("patient_id"),
            "prior_visit_count": len(memory.get("prior_visits") or []),
            "summary": memory.get("summary"),
        },
        "tool_call": final.get("tool_result"),
        "errors": final.get("errors", []),
    }


def generate_note(transcript: str):
    if not transcript or not transcript.strip():
        gr.Warning("Paste or type a transcript before generating a note.")
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            None,
        )

    try:
        final = run_agent(
            transcript,
            patient_id=MOCK_PATIENT_ID,
            patient_age=MOCK_PATIENT_AGE,
            patient_sex=MOCK_PATIENT_SEX,
        )
    except Exception as exc:  # surface the failure in the UI, never a stack trace
        logger.exception("Agent run failed")
        gr.Warning(f"Note generation failed: {exc}")
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            None,
        )

    note = strip_codes_section(final["final_response"])
    # Emergency encounters carry an escalation warning above the SOAP note —
    # lift it into a red alert banner instead of rendering it as plain prose.
    escalation, note = split_escalation(note)
    banner_html = render_emergency_banner(escalation) if escalation else ""
    # Soft failures (zero-hit, stub tools) ride the errors channel — surface
    # them under the note instead of hiding them (skip any already in the note).
    extra = [e for e in final.get("errors", []) if e not in note]
    if extra:
        note += "\n\n---\n" + "\n".join(f"> ⚠️ {e}" for e in extra)

    codes_html = render_code_chips(final.get("suggested_codes") or [])
    return (
        gr.update(visible=False),
        gr.update(value=banner_html, visible=bool(banner_html)),
        gr.update(value=note, visible=True),
        gr.update(value=codes_html, visible=bool(codes_html)),
        gr.update(value=build_trace(final), visible=True),
        final,
    )


def save_to_chart(agent_state: dict | None):
    """Task 11-13: save the just-generated note via the save_note tool.

    Takes the note straight from the agent_state produced by generate_note
    (a gr.State, not a re-run) — the UI is the thing holding "what was just
    reviewed" across the two clicks, since the graph itself keeps no
    conversation memory between calls.
    """
    if not agent_state or not agent_state.get("draft_note"):
        gr.Warning("Generate a note before saving it to the chart.")
        return gr.update(visible=False)

    final = run_agent(
        "Save this note to the patient's chart.",
        patient_id=MOCK_PATIENT_ID,
        note=agent_state["draft_note"],
        suggested_codes=agent_state.get("suggested_codes"),
    )
    tool_result = final.get("tool_result") or {}
    if tool_result.get("ok"):
        gr.Info(tool_result["detail"])
    else:
        gr.Warning(tool_result.get("detail", "Save failed."))
    return gr.update(value=tool_result, visible=True)


def load_patient_history():
    """Task 12 demo: pull the mock patient's prior visits via get_patient_history."""
    from mednote.tools.get_history import get_patient_history

    result = get_patient_history.invoke({"patient_id": MOCK_PATIENT_ID})
    if result["status"] != "found":
        return gr.update(value="No prior visits found for this patient.", visible=True)

    lines = [
        f"**{visit['timestamp'][:10]}** (note `{visit['note_id']}`, "
        f"codes: {', '.join(visit['icd_codes']) or 'none'}):\n\n{visit['note'][:300]}"
        for visit in result["visits"]
    ]
    return gr.update(value="\n\n---\n\n".join(lines), visible=True)


def _warm_up():
    """Preload SapBERT/Qdrant/LLM clients so the first click doesn't pay ~60s."""
    from mednote.agent.nodes import get_note_llm, get_rag_pipeline

    try:
        get_rag_pipeline()
        get_note_llm()
        logger.info("Warm-up complete: RAG stack and LLM ready.")
    except Exception:
        logger.exception(
            "Warm-up failed — is data/qdrant_data built (scripts/build_index.py) "
            "and free of other processes?"
        )


with gr.Blocks(title="MedNote Scribe") as demo:
    with gr.Row(elem_id="app-header"):
        gr.HTML(
            f"""
            <div id="app-title-block">
                <div id="app-logo">{PULSE_ICON}</div>
                <div id="app-titles">
                    <h1>MedNote Scribe</h1>
                    <span>CLINICAL INTELLIGENCE WORKSPACE</span>
                </div>
            </div>
            """
        )
        gr.HTML(
            f"""
            <div id="patient-chip">
                {PERSON_ICON}
                <span class="name">{MOCK_PATIENT_NAME}</span>
                <span class="dob">DOB: {MOCK_PATIENT_DOB} ({MOCK_PATIENT_AGE}y)</span>
            </div>
            """
        )

    with gr.Row():
        with gr.Column(scale=4, elem_classes="panel-card"):
            with gr.Row(elem_classes="panel-header"):
                gr.HTML(f"<h3>{PULSE_ICON_SMALL} Encounter Audio / Transcript</h3>")

            with gr.Row(elem_id="quick-fill-row"):
                routine_btn = gr.Button("Test: Routine", size="sm")
                emergency_btn = gr.Button("Test: Emergency", size="sm")

            transcript_input = gr.Textbox(
                placeholder="Type the encounter transcript here, or click the microphone to start listening...",
                lines=14,
                max_lines=14,
                show_label=False,
                elem_id="transcript-box",
                container=False,
            )

            with gr.Row(elem_id="action-row"):
                dictation_btn = gr.Button("🎙  Start Dictation", elem_id="dictation-btn")
                generate_btn = gr.Button("Generate Note", elem_id="generate-btn")

            with gr.Accordion("🗂  Patient History (EHR)", open=False):
                history_btn = gr.Button("Load Prior Visits", size="sm")
                history_output = gr.Markdown(visible=False)

        with gr.Column(scale=6, elem_classes="panel-card"):
            empty_state = gr.HTML(
                f"""
                <div id="note-empty-state">
                    {DOC_ICON_LARGE}
                    <h3>No Note Generated Yet</h3>
                    <p>Record an encounter or paste a transcript on the left, then click
                    "Generate Note" to structure the data into a SOAP format.</p>
                </div>
                """,
                elem_id="empty-state-wrap",
            )
            emergency_output = gr.HTML(visible=False, elem_id="emergency-output")
            note_output = gr.Markdown(visible=False, elem_id="note-output")
            codes_output = gr.HTML(visible=False, elem_id="codes-output")

            with gr.Row(elem_id="save-row"):
                save_btn = gr.Button("💾  Save to Chart")
            save_status = gr.JSON(visible=False, label="Save Result")

            with gr.Accordion("🔍 Agent Trace", open=False):
                trace_output = gr.JSON(visible=False, label="Execution Trace")

    agent_state = gr.State()

    routine_btn.click(load_routine, outputs=transcript_input)
    emergency_btn.click(load_emergency, outputs=transcript_input)
    dictation_btn.click(start_dictation)
    history_btn.click(load_patient_history, outputs=history_output)
    generate_btn.click(
        generate_note,
        inputs=transcript_input,
        outputs=[
            empty_state,
            emergency_output,
            note_output,
            codes_output,
            trace_output,
            agent_state,
        ],
    )
    save_btn.click(save_to_chart, inputs=agent_state, outputs=save_status)


def main():
    from dotenv import load_dotenv

    load_dotenv(override=True)  # GOOGLE_API_KEY for the two Gemini calls
    threading.Thread(target=_warm_up, daemon=True).start()
    demo.launch(css=CSS)


if __name__ == "__main__":
    main()