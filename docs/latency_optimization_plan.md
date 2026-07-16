# MedNote Latency Optimization Plan

> Status: planned, not yet implemented. Phases are ordered by impact and each is independently shippable.

## Context

Clicking **Generate Note** blocks the Gradio UI for the full pipeline (~10–25 s). The flow is strictly sequential: `app.py:generate_note` → `run_agent` → LangGraph `invoke` with nodes `parse_input → context_extraction → entity_extraction → rag_pipeline → note_generation → guardrail_check → response_generation`. Two sequential Gemini calls dominate (flash-lite entity extraction, then `gemini-pro-latest` note gen, max_tokens=4096, non-streaming). RAG runs a serial per-entity loop (2 Qdrant queries + CPU CrossEncoder rerank each), and the LRU cache only caches pre-rerank candidates, so the reranker runs even on cache hits. `config.yml` declares `demo.latency_budget_ms: 15000` but nothing measures it.

**Key verified insight enabling the structural win**: the RAG codes injected into the note prompt (`src/mednote/agent/nodes.py:187`) feed ONLY the "### Suggested ICD-10 Codes" output section (prompts.py Rule 3); Assessment/Plan are transcript-only (Rule 5); and the UI strips that section and renders chips from graph state instead (`strip_codes_section`, `src/mednote/ui/app.py:298`). So the note LLM only copies codes into text that gets thrown away — meaning note generation and RAG need not be sequential, and the codes section can be assembled deterministically in code (faster AND safer than LLM transcription).

Goal: cut perceived latency to first visible text from ~10–25 s to ~2–4 s and shave total wall time, with zero quality regression (keep gemini-pro, temperature=0, full rerank quality).

## Phases

### Phase 0 — Latency instrumentation (measure first)
- New `src/mednote/observability/latency.py`: `timed_node(name, fn)` wrapper logging trace_id + per-node elapsed ms; total-turn logger that warns when exceeding `demo.latency_budget_ms`.
- `graph.py:build_graph`: wrap each `add_node("x", fn)` with `timed_node`.
- Tests: `tests/test_latency.py` — wrapper preserves return dict, log records emitted (caplog), budget warning fires.
- Impact: none directly; baseline numbers for judging every later phase.

### Phase 1 — Stream the note to the UI (biggest perceived win, zero quality risk)
- `graph.py`: add `stream_agent(...)` generator alongside untouched `run_agent`. Iterate `get_compiled_graph().stream(state, stream_mode=["messages", "values"])`:
  - `"messages"` events where `metadata["langgraph_node"] == "note_generation"` → yield `("token", text)` (filters out the flash-lite extractor's tokens; langchain auto-streams `.invoke()` when a streaming callback is attached, so `nodes.py` needs no change).
  - Track latest `"values"` event; yield `("final", state)` after the loop. Must tolerate zero token events (test fakes don't stream).
- `ui/app.py`: convert `generate_note` to a generator yielding the same 3-tuple. Accumulate tokens, suppress any partial "### Suggested ICD-10 Codes" section, throttle yields (per newline / ~15 tokens). Final yield = today's exact post-processing (`strip_codes_section`, error banner, `render_code_chips`). Keep try/except → warning fallback.
- Tests: update `tests/test_ui_app.py` to consume the generator (`*_, (a, b, c) = ...`; old assertions on last yield unchanged); add partial-yield test; `tests/test_agent_graph.py` adds a `stream_agent` round-trip asserting parity with `run_agent` on non-streaming fakes.
- Impact: first visible text ~10–25 s → ~3–6 s (entity + RAG + note TTFT). Total unchanged.

### Phase 2 — Run note_generation in parallel with rag_pipeline (eval-gated)
- `prompts.py`: SOAP prompt variant without the codes output section / Rule 3 / `{rag_context}` (all other rules verbatim). New `format_codes_section(codes)` producing the exact `[CODE] - [Desc] (Source: ICD-10-CM 2026) (Pending Physician Confirmation)` lines, or the pinned insufficient-data sentence for `[]`.
- `nodes.py`: `note_generation` stops reading `suggested_codes`; `response_generation` (soap branch) appends `format_codes_section(codes)` to the draft note — preserves the `final_response` contract asserted in `test_agent_graph.py:134-137`.
- `graph.py`: fan out after `entity_extraction` via list-valued conditional edges: `soap → ["rag_pipeline", "note_generation"]`, `icd_lookup → "rag_pipeline"`. `rag_pipeline → response_generation` becomes unconditional. Register `response_generation` with `defer=True` (langgraph 1.2.9, locked, supports it) so it joins both branches. State is parallel-safe: branches write disjoint keys; `errors` already has an `operator.add` reducer.
- Streaming synergy: when a `values` event first contains `suggested_codes`, yield `("codes", codes)` so ICD chips render while the note is still streaming.
- **Quality gate before merge**: A/B old vs new prompt over the soap rows of `data/transcripts/synthetic_transcripts.json`; diff S/O/A/P sections; verify the emergency transcript still leads with URGENT ESCALATION. Fallback config flag `llm.parallel_note: false` selecting the old topology if wording regresses.
- Tests: canned note drops its codes section; assert all 5 headers still in `final_response`, codes section exactly once, note-LLM prompt contains no code lines, RAG failure doesn't block the note branch; update `tests/test_prompts.py` pinned contracts + `format_codes_section` tests.
- Impact: total = entity + max(RAG, note LLM) instead of entity + RAG + note. Combined with Phase 1, first tokens ~2–4 s.

### Phase 3 — Parallel + batched RAG internals
- `rag/pipeline.py:_retrieve_and_rerank`: two stages —
  1. Candidate fetch via `ThreadPoolExecutor` (new config `vector_store.max_parallel_queries`, default 4, `1` = serial kill-switch). Embedded Qdrant local mode is read-only at serve time (index built offline) so concurrent reads are safe; kill-switch is insurance.
  2. Batched rerank: new `ClinicalReranker.rerank_many(items, top_n)` — one flat `CrossEncoder.predict` across all entities, scores split back by offsets; bit-identical to per-entity calls (per-pair independence), so zero quality change. `rerank()` delegates to it.
- `rag/cache.py`: add `threading.Lock` around get/set + counters.
- `rag/specificity.py`: batch child-code fetches — one `scroll` with `MatchAny(all_codes)` instead of one per parent.
- Optional micro-win: batch-embed all entities in one SapBERT forward pass.
- Tests: multi-entity output identical to serial path; `rerank_many` ≡ N × `rerank` (FakeCrossEncoder); cache thread-safety hammer; `max_parallel_queries: 1` fallback.
- Impact: ~2–3× RAG wall time reduction for typical 3-entity notes (~0.5–1.5 s warm); speeds `icd_lookup` intent and how early chips appear.

### Phase 4 — Post-rerank cache
- Second cache layer in the existing `RAGCache` keyed `rerank|{entity}|sex|age|k` storing reranked results; hit skips retrieval AND rerank. Existing pre-rerank cache stays as fallback layer. No invalidation needed (index immutable per process).
- Tests: second identical run makes zero retriever/reranker calls (counting fakes); demographic change misses; LRU bound holds.
- Impact: repeated entities (common in a clinic day) drop from ~300–800 ms to <5 ms.

## Explicitly deferred
- Gemini thinking-budget tuning: likely the largest total-latency lever but a direct quality risk — separate eval-gated experiment only.
- `max_tokens=4096`: leave as-is (uncapped-unless-hit; lowering risks truncation).
- Dense+sparse intra-entity parallelism: covered by entity-level parallelism.

## Critical files
- `src/mednote/ui/app.py` — generator handler, streaming display, early chips
- `src/mednote/agent/graph.py` — `stream_agent`, fan-out edges, `defer=True` join, timing wrappers
- `src/mednote/agent/nodes.py` — note prompt decoupling, deterministic codes assembly
- `src/mednote/agent/prompts.py` — prompt variant, `format_codes_section`
- `src/mednote/rag/pipeline.py`, `rag/reranker.py`, `rag/cache.py`, `rag/specificity.py` — parallel/batched RAG, post-rerank cache
- `src/mednote/observability/latency.py` (new), `config.yml` (new keys)

## Verification
1. Phase 0 logs give per-node baseline on `Test: Routine` and `Test: Emergency` transcripts.
2. After each phase: `pytest tests/` green; run the app (`Generate Note` on both test transcripts) and confirm identical final note/chips vs baseline, plus the latency log deltas.
3. Phase 1: visually confirm token-by-token rendering and that the codes section never flashes in the note pane.
4. Phase 2 gate: A/B S/O/A/P diff over synthetic transcripts + emergency escalation check before merge.
5. Compare total-turn timing against the 15 s budget warning.

## Expected cumulative outcome
| After | First visible text | Total wall time |
|---|---|---|
| Baseline | ~10–25 s | ~10–25 s |
| P1 streaming | ~3–6 s | unchanged |
| P2 parallel graph | ~2–4 s | −0.5–2 s |
| P3 RAG parallel | unchanged | chips/icd_lookup −0.5–1.5 s |
| P4 rerank cache | unchanged | repeat queries ≈ 0 RAG cost |
