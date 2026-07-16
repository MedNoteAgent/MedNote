# MedNote Scribe

MedNote Scribe is a clinical documentation agent for outpatient physicians. It converts doctor–patient transcripts into draft **SOAP notes**, suggests **ICD-10-CM codes** grounded in a retrieval pipeline (never from the model's own memory), and flags **red-flag emergencies** for urgent escalation. Every output is a draft pending physician confirmation — the agent never asserts a diagnosis.

## What's implemented

- **LangGraph agent** — a 9-node graph routing five intents: SOAP note generation, ICD-10 lookup, save note, visit history, and refusal (e.g. "diagnose this patient"). Tool execution and memory are stubs until Week 2.
- **ICD-10 RAG pipeline** — 46,887 ICD-10-CM 2026 codes + clinical guideline sections indexed in embedded Qdrant. Retrieval is hybrid: SapBERT dense vectors (clinical paraphrases) + BM25 sparse vectors (exact terms), fused and reranked by a cross-encoder, with demographic hard filters (sex/age-impossible codes are excluded, not down-ranked) and hierarchical specificity expansion.
- **Entity extraction** — a fast LLM rewrites colloquial transcript language into formal clinical terms before retrieval, restricted to findings documented as present: symptoms a clinician merely asks about are ignored, and symptoms are never upgraded to inferred diagnoses.
- **Safety-first prompts** — hedged assessments only, no invented dosages, codes cited from retrieved context or an explicit "assign manually in EHR" zero-hit message, prompt-injection defense (the transcript is data, not instructions).
- **Gradio UI** — transcript input with routine/emergency presets, draft note, ICD-10 code chips with confidence tooltips and physician-confirmation checkboxes, and a red escalation banner that bolds red-flag symptoms on emergency encounters.

## How it works

```
transcript ─► intent router ─► entity extraction ─► hybrid retrieval ─► rerank
                                (fast LLM)           (Qdrant: dense+sparse)
           ─► SOAP generation (main LLM, RAG codes injected) ─► guardrails ─► UI
```

## Setup

### 1. Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- An LLM API key — Google Gemini by default (`config.yml`); Anthropic and OpenAI also supported

### 2. Install

```bash
uv sync            # runtime dependencies
uv sync --extra dev  # + test/lint tooling
```

### 3. Configure secrets

```bash
cp .env.example .env   # then fill in GOOGLE_API_KEY (or your provider's key)
```

All tunables (models, RAG weights, paths, thresholds) live in `config.yml`; `.env` holds secrets only.

### 4. Build the RAG index (one-time)

```bash
uv run python scripts/download_icd10.py   # fetch ICD-10-CM 2026 XML from CMS.gov (idempotent)
uv run python scripts/run_etl.py          # XML -> enriched JSONL code documents
uv run python scripts/build_index.py      # embed + upsert into embedded Qdrant
uv run python scripts/validate_index.py   # sanity-check counts and retrieval quality
```

Background: the ETL parses the official CMS release into ~47k self-contained code documents (descriptions, synonyms, hierarchy, demographic applicability). The build step embeds each with SapBERT (dense) and BM25 (sparse) into one hybrid collection on local disk — **~70 minutes on CPU**; it is resumable, so re-running skips already-indexed points.

> **Note:** embedded Qdrant is single-process. Finish the index build (and close any notebooks using it) before starting the app.

### 5. Run the app

```bash
uv run python -m mednote.ui.app
```

Allow ~60s of warm-up (SapBERT + Qdrant load) after launch. Use the **Test: Routine** / **Test: Emergency** buttons for instant demo transcripts.

### 6. Run the tests

```bash
uv run pytest
```

## Project layout

```
src/mednote/agent/   LangGraph nodes, prompts, state, graph
src/mednote/rag/     ETL, indexer, retriever, reranker, pipeline
src/mednote/llm/     provider-agnostic LLM wrapper
src/mednote/ui/      Gradio app
scripts/             download / ETL / index build / validation
data/                corpus, processed codes, Qdrant store, transcripts
docs/                requirements, implementation plan, design notes
```

See `docs/requirements.md` for objectives and guardrail requirements, and `docs/implementation_plan.md` for the full task roadmap.
