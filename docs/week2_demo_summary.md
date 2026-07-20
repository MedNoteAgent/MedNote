# MedNote Scribe — Week 2 Summary (Tasks 10-16)

**Goal of Week 2:** Tools, MCP & Memory — the same Gradio UI now saves a note
through the mock EHR tool and recalls a patient's prior-visit context, visible
live in the chat.

## What we built

### Tool specs (Task 10)
- `docs/tools.md` — the `save_note` / `get_patient_history` contract (inputs,
  outputs, error cases) both the LangChain tool wrappers and the MCP server
  implement identically.

### Mock EHR + `save_note` (Task 11)
- `EHRStore` (`src/mednote/tools/ehr_api.py`): thread-safe, JSON-file-backed
  (`data/ehr_store.json`), with `save_note()` / `get_history()`.
- A FastAPI app wraps the same store as an independently runnable REST
  server (`POST /notes`, `GET /patients/{id}/history`) — the "realistic REST
  surface" `requirements.md` §4 asks for.
- **Deviation from the plan's sketch:** the LangChain tool and the MCP server
  call `EHRStore` in-process rather than over `httpx`, since both run in the
  same process for this demo. Documented in `docs/tools.md` "Implementation
  notes," in the same spirit as the Week 1 retrieval/indexing deviation notes.

### `get_patient_history` (Task 12)
- Second tool wrapper over the same store; `"found"` (visits newest first)
  or `"no_history"` for a new patient — never an exception for the "no prior
  visits" case, since that's an expected, common outcome.

### MCP server (Task 13)
- `src/mednote/mcp/server.py` exposes both tools via `FastMCP`, which derives
  the tool schema from each function's signature and docstring — one
  definition of the contract, not a hand-written schema alongside the code.
- Runnable standalone over stdio (`uv run python -m mednote.mcp.server`) for
  any general-purpose MCP client; tested via `FastMCP.call_tool()` directly
  (`tests/test_mcp.py`) for a real round trip without a subprocess.

### Memory schema (Task 14)
- `MemoryStore` (`src/mednote/memory/store.py`): one SQLite table, `visits`
  (`patient_id`, `visit_date`, `note_id`, `summary`, `icd_codes`), separate
  from the EHR's JSON store — the EHR is the system of record; memory is the
  agent's own recall index, shaped for prompt injection rather than charting.

### Memory integration (Task 15)
- `tool_execution` writes to both stores on a successful save: the EHR record
  and a mirrored `MemoryStore` visit summary.
- `memory_lookup` (the "history" intent) and `context_extraction` (the SOAP
  path, so a returning patient's new note gets continuity) both read through
  a shared `_build_memory_context()` helper.
- `note_generation`'s prompt now carries a "Patient context from prior
  visits" section (`SOAP_USER_PROMPT`'s new `{memory_context}` slot), framed
  explicitly as continuity, not a new finding.
- Verified across two independent `run_agent()` calls (no shared graph
  state) in `tests/test_agent_graph.py::test_history_recalls_note_saved_in_a_prior_session`
  — a fact from a "session 1" save is recalled, unprompted, in "session 2."

### Agent trace panel + UI wiring (Task 16)
- `build_trace()` renders intent, extracted entities, RAG retrieval (code /
  confidence / source), cache-hit, memory used (prior-visit count + summary),
  and the tool-call result into a `gr.JSON` accordion.
- New "Save to Chart" button calls `save_note` with the note the UI just
  displayed (held in a `gr.State`, since the graph itself keeps no
  conversation memory between clicks) and reports the result.
- New "Patient History" accordion calls `get_patient_history` directly to
  show prior visits for the demo patient before generating a new note.
- Node-by-node timing is deliberately deferred to Week 4 (Task 24
  observability) — this panel surfaces what Week 2's nodes actually produce.

## By the numbers
- 30 new automated tests (tools, memory, MCP, graph save/history paths, UI)
- 143 tests passing, 1 skipped (up from 114 at Week 1)
- 2 new SQLite/JSON-backed stores, 2 LangChain tools, 1 MCP server, 3 new UI
  controls (Save to Chart, Patient History, Agent Trace)

## Suggested demo flow
1. **Generate a note** for the demo patient (routine transcript) → note +
   ICD-10 chips, same as Week 1.
2. **Expand "Agent Trace"** → show the RAG retrieval, memory context, and
   (before saving) the empty tool-call slot.
3. **Click "Save to Chart"** → success message with a real `note_id`; expand
   "Patient History" → the just-saved visit appears.
4. **Generate a second note** for the same patient → the trace panel's
   `memory_used.summary` shows the prior visit, and the note's context
   (invisible to the user, visible in the prompt) carries it forward —
   demonstrating recall across sessions without re-pasting history.

## Next (Week 3 preview)
Deterministic guardrails (red-flag detection, dosage-fabrication check,
diagnosis-assertion reframing) and RAG caching with a visible cache-hit badge
(Tasks 17-23).
