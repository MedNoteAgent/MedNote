# MedNote Scribe — Week 1 Summary (Tasks 1–9)

**Goal of Week 1:** Foundations, RAG & UI — a working end-to-end prototype that turns a
doctor–patient transcript into a draft SOAP note with grounded ICD-10-CM code suggestions.

## What we built

### Foundations (Tasks 1, 1B, 2, 2B)
- Project scaffolding with `uv`, requirements and a 32-task implementation plan.
- Central `config.yml` — every model name, path, weight and threshold lives in config, not code.
- Unified LLM wrapper supporting OpenAI / Anthropic / Google via LangChain, with env-var
  overrides and a separate fast/cheap model tier for lightweight calls.

### Safety-first prompt library (Task 3)
- SOAP system prompt with pinned section structure and hard clinical rules: never diagnose,
  never invent dosages, hedged assessment language, red-flag escalation criteria, and
  prompt-injection defense (the transcript is data, not instructions).
- Escalation, refusal, and ICD-lookup prompts; codes may come **only** from retrieved
  reference context, with an explicit "insufficient data" behavior instead of guessing.
- Contract tests pin every safety rule so a prompt edit that drops one fails CI.

### Evaluation dataset (Task 4)
- 19 synthetic labeled transcripts covering routine, emergency, and edge cases — each with
  expected intent, red-flag label, and expected ICD-10 codes for the Week 4 eval harness.

### ICD-10-CM knowledge base (Tasks 5–6)
- ETL pipeline (`scripts/run_etl.py`) that downloads and parses the official ICD-10-CM 2026
  release plus a clinical guidelines corpus.
- **46,887 documents indexed** into embedded Qdrant: SapBERT dense vectors (clinical
  paraphrase matching) + BM25 sparse vectors (exact terms/acronyms), with demographic
  metadata (sex/age applicability) and parent/child hierarchy links in payloads.

### Retrieval pipeline (Task 7)
- LLM entity extraction rewrites colloquial transcript language into formal clinical terms —
  restricted to findings documented as present (a clinician's screening question is not a
  finding; symptoms are never upgraded to inferred diagnoses).
- Hybrid dense+sparse retrieval with demographic **hard filters** (impossible codes excluded,
  not down-ranked), cross-encoder reranking per entity, per-entity fairness in the merge,
  hierarchical specificity expansion, LRU caching, and a zero-hit protocol that says
  "assign manually in EHR" instead of guessing.

### Agent + UI (Tasks 8–9)
- LangGraph agent: 9 nodes, intent router (SOAP / ICD lookup / save / history / refuse),
  RAG pipeline, note generation, guardrail hook (deterministic rules land in Week 3),
  and response formatting. Thread-safe service initialization.
- Gradio UI: transcript input with routine/emergency test presets, draft SOAP note,
  ICD-10 code chips with hover descriptions and physician-confirmation checkboxes, and a
  bold red **emergency escalation banner** that highlights red-flag risks for
  emergency encounters.

## By the numbers
- 46,887 indexed ICD-10 + guideline documents
- 19 labeled evaluation transcripts
- 114 automated tests passing
- 9-node agent graph routing 5 intents
- 8 design/process docs + 2 walkthrough notebooks

## Suggested demo flow
1. **Routine encounter** — click *Test: Routine* → *Generate Note*: structured SOAP draft,
   code chips with confidence tooltips, everything marked *Pending Physician Confirmation*.
2. **Emergency encounter** — *Test: Emergency*: red escalation banner names the triggering
   red-flag symptoms in bold; symptom-level codes (not premature cardiac diagnoses).
3. **Safety behaviors** — type "Diagnose this patient" → refusal; a too-short transcript →
   graceful insufficient-input message instead of a hallucinated note.
4. **Under the hood** — `docs/notebooks/agent_graph.ipynb` for the graph, and the
   retrieval notes for how a transcript becomes grounded codes.

## Next (Week 2 preview)
Mock EHR + `save_note` / `get_patient_history` tools over MCP, visit memory, and the
agent trace panel in the UI (Tasks 10–16).
