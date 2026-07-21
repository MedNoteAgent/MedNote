"""Gradio UI for MedNote Scribe (tasks.md Tasks 9 + 16).

"Generate Note" runs the full LangGraph agent (intent routing -> RAG over the
ICD-10 index -> SOAP generation -> guardrail stub) via run_agent(). Task 16
adds: a "Save to EHR" button (explicit physician save -> tool_execution's LLM
tool call), a Prior Visits accordion fed by SQLite visit memory, and an Agent
Trace accordion rendered from the final agent state (the per-node tracer
arrives with Task 24). "Start Dictation" remains a stub since live audio
transcription is out of scope for the demo (requirements.md §4).

Launch (from the repo root, after `scripts/build_index.py` has been run):
    uv run uvicorn mednote.tools.ehr_api:app --port 8100   # terminal 1 (for Save)
    uv run python -m mednote.ui.app                        # terminal 2
Without the EHR server, note generation still works; Save degrades to a
clear "NOT saved" banner.
"""

import html
import logging
import re
import threading
import time

import gradio as gr

from mednote.agent.graph import run_agent

logger = logging.getLogger(__name__)

# Demo patient shown in the header chip; demographics feed the RAG demographic
# filter (seeded in the mock EHR as P-DEMO with matching age/sex).
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
#save-row {
    padding: 0 20px 16px 20px !important;
}
#save-btn {
    background: #1e7e46 !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}
#save-status {
    padding: 0 20px 16px 20px !important;
}
.save-status-banner {
    border-radius: 10px;
    padding: 12px 16px;
    font-weight: 600;
    font-size: 0.9rem;
}
.save-status-banner.success {
    background: #e8f5ec;
    border: 1px solid #b7e0c3;
    color: #1e7e46;
}
.save-status-banner.failure {
    background: #fdf6e3;
    border: 1px solid #f0d9a0;
    color: #8a6d1a;
}
#trace-accordion, #history-accordion {
    margin: 0 20px 20px 20px !important;
    border: 1px solid #e5e9ef !important;
    border-radius: 10px !important;
    background: #fafbfc !important;
}
/* Gradio emits light text when the OS is in dark mode — the container forces
   a light background, so every text inside the new panels must be pinned
   dark explicitly (same reason #note-output pins its colors above). */
#trace-accordion .label-wrap, #history-accordion .label-wrap,
#trace-accordion .label-wrap span, #history-accordion .label-wrap span {
    color: #1a2233 !important;
    font-weight: 600;
}
#history-accordion .prose, #history-accordion .prose * {
    font-size: 0.85rem;
    color: #2a3346 !important;
}
#history-refresh-btn {
    background: white !important;
    border: 1px solid #d7dce4 !important;
    color: #3a4356 !important;
    border-radius: 999px !important;
    font-size: 0.8rem !important;
    box-shadow: none !important;
}
/* gr.JSON (and gr.Markdown) wrap themselves in .block containers whose
   background comes from the theme (dark in OS dark mode) — flatten every
   wrapper inside the accordions so content sits on the light panel. */
#trace-accordion .block, #trace-accordion .json-holder, #trace-accordion .container,
#history-accordion .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#trace-accordion .json-holder, #trace-accordion .json-holder * {
    color: #2a3346 !important;
    background: transparent !important;
}
.codes-footer, .codes-footer em, .codes-footer strong {
    color: #6b7488 !important;
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


def build_trace(final: dict, elapsed_ms: float) -> dict:
    """Compact execution trace rendered from the final agent state (Task 16).

    Per-node timing and the raw pre-rerank candidates arrive with the real
    tracer (Task 24) — this deliberately reads only what the graph already
    returns, so there is no second tracing mechanism to keep in sync.
    """
    return {
        "trace_id": final.get("trace_id"),
        "intent": final.get("intent"),
        "latency_ms": round(elapsed_ms),
        "extracted_entities": final.get("extracted_entities") or [],
        "suggested_codes": [
            {"code": c.get("code"), "confidence": round(c.get("confidence", 0.0), 3)}
            for c in (final.get("suggested_codes") or [])
        ],
        "cache_hit": final.get("cache_hit", False),
        "guardrail": final.get("guardrail_result"),
        "tool_result": final.get("tool_result"),
        "memory": (final.get("memory_context") or {}).get("summary"),
        "errors": final.get("errors") or [],
    }


def fetch_memory_context(patient_id: str) -> dict | None:
    """Prior-visit context for SOAP continuity (Task 15). None when empty,
    so first visits add nothing to the prompt."""
    from mednote.agent.nodes import memory_lookup

    context = memory_lookup({"patient_id": patient_id})["memory_context"]
    return context if context["prior_visits"] else None


def render_prior_visits(patient_id: str = MOCK_PATIENT_ID) -> str:
    """Markdown for the Prior Visits accordion — the same SQLite memory the
    history intent reads (free: no LLM, no EHR server needed)."""
    from mednote.agent.nodes import memory_lookup

    return memory_lookup({"patient_id": patient_id})["memory_context"]["summary"]


def render_save_status(tool_result: dict) -> str:
    if tool_result.get("ok"):
        note_id = html.escape(str(tool_result.get("note_id")))
        return (
            '<div class="save-status-banner success">✅ Saved to EHR — note ID '
            f"{note_id}. Also recorded in visit memory.</div>"
        )
    detail = html.escape(tool_result.get("detail", "Save failed."))
    return f'<div class="save-status-banner failure">⚠️ {detail}</div>'


def _generation_failed():
    """The 8-output tuple for paths where no note was produced."""
    hidden = gr.update(visible=False)
    return (
        gr.update(visible=True),   # empty state back
        hidden,                    # emergency banner
        hidden,                    # note
        hidden,                    # code chips
        hidden,                    # save button
        hidden,                    # save status
        gr.update(),               # trace unchanged
        None,                      # last_result state cleared
    )


def generate_note(transcript: str):
    if not transcript or not transcript.strip():
        gr.Warning("Paste or type a transcript before generating a note.")
        return _generation_failed()

    t0 = time.perf_counter()
    try:
        final = run_agent(
            transcript,
            patient_id=MOCK_PATIENT_ID,
            patient_age=MOCK_PATIENT_AGE,
            patient_sex=MOCK_PATIENT_SEX,
            # Task 15: prior-visit continuity flows into SOAP generation.
            memory_context=fetch_memory_context(MOCK_PATIENT_ID),
        )
    except Exception as exc:  # surface the failure in the UI, never a stack trace
        logger.exception("Agent run failed")
        gr.Warning(f"Note generation failed: {exc}")
        return _generation_failed()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    note = strip_codes_section(final["final_response"])
    # Emergency encounters carry an escalation warning above the SOAP note —
    # lift it into a red alert banner instead of rendering it as plain prose.
    escalation, note = split_escalation(note)
    banner_html = render_emergency_banner(escalation) if escalation else ""
    # Soft failures (zero-hit, EHR down) ride the errors channel — surface
    # them under the note instead of hiding them (skip any already in the note).
    extra = [e for e in final.get("errors", []) if e not in note]
    if extra:
        note += "\n\n---\n" + "\n".join(f"> ⚠️ {e}" for e in extra)

    codes = final.get("suggested_codes") or []
    codes_html = render_code_chips(codes)
    # What the Save button will hand to tool_execution (guardrail G5: the
    # save happens only when the physician clicks — this is just staging).
    last_result = {
        "patient_id": MOCK_PATIENT_ID,
        "note": final.get("draft_note") or final["final_response"],
        "codes": codes,
    }
    return (
        gr.update(visible=False),
        gr.update(value=banner_html, visible=bool(banner_html)),
        gr.update(value=note, visible=True),
        gr.update(value=codes_html, visible=bool(codes_html)),
        gr.update(visible=True),                       # reveal Save to EHR
        gr.update(value="", visible=False),            # reset old save status
        gr.update(value=build_trace(final, elapsed_ms)),
        last_result,
    )


def save_note_to_ehr(last_result: dict | None):
    """Explicit physician save (guardrail G5) — routes the save intent through
    the agent, so tool_execution's LLM makes the actual tool call. 💰 1 LLM
    call; needs the mock EHR running (degrades to a failure banner if not)."""
    if not last_result or not last_result.get("note"):
        gr.Warning("Generate a note first — there is nothing to save.")
        return gr.update(visible=False), gr.update(), gr.update()

    t0 = time.perf_counter()
    try:
        final = run_agent(
            "Save this note to the patient's chart.",
            patient_id=last_result["patient_id"],
            draft_note=last_result["note"],
            suggested_codes=last_result["codes"],
        )
    except Exception as exc:
        logger.exception("Save to EHR failed")
        status = render_save_status({"ok": False, "detail": f"Save failed: {exc}"})
        return gr.update(value=status, visible=True), gr.update(), gr.update()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    tool_result = final.get("tool_result") or {"ok": False, "detail": "No tool result returned."}
    return (
        gr.update(value=render_save_status(tool_result), visible=True),
        gr.update(value=build_trace(final, elapsed_ms)),
        gr.update(value=render_prior_visits(last_result["patient_id"])),
    )


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

            with gr.Accordion(
                "🗂 Prior Visits (agent memory)", open=False, elem_id="history-accordion"
            ):
                history_output = gr.Markdown(
                    "Click *Refresh* to load this patient's prior visits from memory."
                )
                refresh_history_btn = gr.Button(
                    "Refresh", size="sm", elem_id="history-refresh-btn"
                )

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

            with gr.Row(elem_id="save-row", visible=False) as save_row:
                save_btn = gr.Button("💾 Save to EHR", elem_id="save-btn")
            save_status = gr.HTML(visible=False, elem_id="save-status")

            with gr.Accordion("🔍 Agent Trace", open=False, elem_id="trace-accordion"):
                trace_output = gr.JSON(value=None)

    last_result = gr.State(None)

    routine_btn.click(load_routine, outputs=transcript_input)
    emergency_btn.click(load_emergency, outputs=transcript_input)
    dictation_btn.click(start_dictation)
    generate_btn.click(
        generate_note,
        inputs=transcript_input,
        outputs=[
            empty_state, emergency_output, note_output, codes_output,
            save_row, save_status, trace_output, last_result,
        ],
    )
    save_btn.click(
        save_note_to_ehr,
        inputs=last_result,
        outputs=[save_status, trace_output, history_output],
    )
    refresh_history_btn.click(render_prior_visits, outputs=history_output)


def main():
    from dotenv import load_dotenv

    load_dotenv(override=True)  # GOOGLE_API_KEY for the two Gemini calls
    threading.Thread(target=_warm_up, daemon=True).start()
    demo.launch(css=CSS)


if __name__ == "__main__":
    main()