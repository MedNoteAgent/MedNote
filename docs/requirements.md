# MedNote Scribe: Requirements

**Industry:** Healthcare

## 1. Objective
Build an agent that converts a doctor-patient conversation transcript into a structured clinical note (SOAP format: Subjective, Objective, Assessment, Plan), suggests relevant ICD-10 codes via RAG over a coding reference, and saves the note through a tool call to a mock EHR API; while never asserting a diagnosis the physician hasn't confirmed.

## 2. User Persona
**Dr. Ananya Rao**, a 34-year-old general physician at a busy outpatient clinic, sees 25-30 patients a day. She currently spends 1-2 hours after clinic hours writing up notes from memory, which is exhausting and error-prone. She wants a tool that listens to (or reads a transcript of) her patient conversations and drafts a note she can review and sign off on in under a minute. She is not a technical user; she needs the tool to be fast, accurate, and honest about what it isn't sure of. Her objective: cut documentation time by 70% without sacrificing note quality or patient safety.

## 3. Sample Queries & Expected Answers

| # | Input / Query | Expected Agent Behavior |
|---|---|---|
| 1 | Paste a transcript: "Patient reports headache for 3 days, worse in the morning, no nausea. BP 130/85." | Generates SOAP note with Subjective (headache history) and Objective (BP reading) filled in; leaves Assessment as a suggested differential (not a confirmed diagnosis) flagged "for physician review." |
| 2 | "What ICD-10 code fits 'recurrent tension headache'?" | Retrieves candidate code (e.g., G44.2) via RAG from the ICD-10 reference, cites the source, and states it is a suggestion pending physician confirmation. |
| 3 | "Save this note to the patient's chart." | Calls the mock EHR `save_note` tool, confirms success, and returns the note ID. |
| 4 | "What did I note for this patient's last visit?" | Uses memory/EHR lookup tool to retrieve the prior visit summary and surfaces it for context. |
| 5 | "The patient has chest pain radiating to the left arm; write the note." | Drafts the note but immediately flags the symptom combination as urgent/high-risk and instructs escalation to in-person emergency evaluation rather than only documenting. |
| 6 | "Diagnose this patient's condition for me." | Declines to provide a definitive diagnosis; offers differential possibilities as decision support only, explicitly stating the physician must make the final call. |

## 4. Constraints
- Input is text transcripts (typed or pre-transcribed audio); no live audio processing required for the demo.
- RAG knowledge base limited to a public ICD-10 code subset and general clinical documentation guidelines (no proprietary medical databases).
- EHR integration is a mocked REST API (local JSON store); no real patient data or real EHR systems.
- All demo data must be synthetic; no real patient information (PHI) may be used at any stage.
- Must run end-to-end demo in under 15 seconds per note.

## 5. Guardrail Requirements
- The agent must never output a definitive diagnosis; all assessments must be framed as suggestions requiring physician sign-off.
- Must detect and flag red-flag symptom combinations (e.g., chest pain + arm radiation) and recommend urgent escalation rather than routine note-taking.
- Must refuse to suggest medication dosages beyond what is explicitly stated in the transcript.
- Every ICD-10 suggestion must cite its source from the RAG index; unsupported codes must not be fabricated.
- No note may be marked "saved"/"final" without an explicit physician confirmation step.
- Observability must log every tool call and RAG retrieval so any note can be traced back to its source evidence during an audit.