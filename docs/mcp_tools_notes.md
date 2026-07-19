# MCP & EHR Tooling — How It Works (Tasks 10–13)

This note explains the tool system built on `ft_mcp_build`: the mock EHR server,
the two clinical tools exposed over **MCP (Model Context Protocol)** and
**LangChain**, and how a physician's "save this note" request travels through
the agent to the EHR and back.

> **Hands-on companion:** [`docs/notebooks/mcp_tools.ipynb`](notebooks/mcp_tools.ipynb)
> walks every layer live, from raw HTTP up to a real MCP stdio handshake.

---

## 1. The Big Picture

The design rule is **one implementation, many surfaces**. There is exactly one
piece of code that knows how to talk to the EHR — `ehr_client.py` — and every
consumer (the LangGraph agent, any MCP client, tests) goes through it. The two
tool surfaces are one-line delegates, so their behavior can never drift apart.

```
                 CONSUMERS                        SHARED CORE                 BACKEND
┌─────────────────────────────────────┐   ┌───────────────────────┐   ┌──────────────────────┐
│                                     │   │                       │   │                      │
│  LangGraph agent                    │   │  tools/ehr_client.py  │   │  tools/ehr_api.py    │
│  ├─ tool_execution node             │   │                       │   │  (FastAPI, port 8100)│
│  │   └─ LLM.bind_tools([...])       │──►│  post_note()          │──►│  POST /notes         │
│  │      ├─ tools/save_note.py       │   │  fetch_history()      │   │  GET  /patients/{id} │
│  │      └─ tools/get_history.py     │   │  fetch_demographics() │   │       /history       │
│  └─ context_extraction node ────────│──►│                       │   │  GET  /patients/{id} │
│                                     │   │  raises EhrApiError   │   │                      │
│  MCP clients (Claude Desktop,       │   │  when EHR unreachable │   │  JSON file store     │
│  Claude Code, any MCP host)         │   │                       │   │  data/ehr_store.json │
│  └─ mcp/server.py (FastMCP, stdio) ─│──►│                       │   │  (seeded P001–P013   │
│      ├─ @mcp.tool save_note         │   │                       │   │   + P-DEMO)          │
│      └─ @mcp.tool get_patient_      │   └───────────────────────┘   └──────────────────────┘
│         history                     │
└─────────────────────────────────────┘
```

Key point: the **MCP server is a peer of the LangGraph agent**, not part of it.
The agent uses the LangChain `@tool` wrappers directly (in-process); the MCP
server exposes the *same* operations to external MCP hosts over stdio. Both
call the same `ehr_client` functions, which make HTTP calls to the FastAPI EHR.

---

## 2. The Components

### 2.1 Mock EHR server — `src/mednote/tools/ehr_api.py`

A FastAPI app standing in for a real EHR (per requirements §4: mock only).

| Endpoint | Purpose | Success envelope |
|----------|---------|------------------|
| `POST /notes` | Save a note (validates `patient_id`, non-empty `note`; optional `icd_codes`) | `{"status": "saved", "note_id": "N_xxxxxxxx", "patient_id", "timestamp"}` |
| `GET /patients/{id}/history` | Prior visits for a patient | `{"status": "found", "visits": [...]}` or `{"status": "no_history", ...}` |
| `GET /patients/{id}` | Demographics (name, age, sex) | `{"status": "found", "patient": {...}}` |

Failures use a consistent error envelope `{"status": "error", "message": ...}`
(404 for unknown patients, 422 for invalid payloads). State lives in
`data/ehr_store.json` (gitignored — runtime-mutated), created on first write
from `SEED_PATIENTS`: the P001–P013 patients of the synthetic transcript set
plus the UI's demo patient `P-DEMO`, so demographics always match the eval
labels.

Run it: `uv run uvicorn mednote.tools.ehr_api:app --port 8100`
(host/port come from `config.yml → ehr_api`).

### 2.2 Shared HTTP core — `src/mednote/tools/ehr_client.py`

Three functions, one per endpoint: `post_note`, `fetch_history`,
`fetch_demographics`. Each builds an `httpx.Client` via `get_client()` — the
**test seam**: tests monkeypatch `get_client` to return a FastAPI `TestClient`,
so every tool test is a full HTTP round-trip against the real app with a
temp-file store, no server process and no network.

Any connection or protocol failure raises `EhrApiError`. Callers are required
to turn that into a graceful degradation — never a stack trace.

### 2.3 LangChain tools — `save_note.py`, `get_history.py`

`@tool`-decorated wrappers whose **signatures and docstrings are the LLM-facing
contract** (LangChain derives the tool schema from them). `save_note`'s
docstring carries the guardrail rule: only call after the physician explicitly
asked to save.

### 2.4 MCP server — `src/mednote/mcp/server.py`

Built with **FastMCP** from the official `mcp` SDK (1.28.1). Note: the
low-level `mcp.server.Server` class has *no* `@tool` decorator — the original
plan sketch used one and would have failed on import; FastMCP is the supported
decorator API.

```python
mcp = FastMCP("mednote-ehr")

@mcp.tool()
def save_note(patient_id: str, note: str, icd_codes: list[str] | None = None) -> dict: ...

@mcp.tool()
def get_patient_history(patient_id: str) -> dict: ...
```

FastMCP auto-generates each tool's JSON schema from the type hints and
docstring, speaks the MCP handshake (`initialize`, `tools/list`, `tools/call`),
and runs over **stdio transport**: an MCP host launches the process and
exchanges JSON-RPC messages over stdin/stdout.

Run it standalone: `uv run python -m mednote.mcp.server`
Register in an MCP host (e.g. Claude Code): `claude mcp add mednote-ehr -- uv run python -m mednote.mcp.server`

---

## 3. Integrated Tools

| Tool | Args | Backing call | Returns |
|------|------|--------------|---------|
| `save_note` | `patient_id` (req), `note` (req), `icd_codes` (opt) | `POST /notes` | `note_id` on success |
| `get_patient_history` | `patient_id` (req) | `GET /patients/{id}/history` | prior visits / no-history message |

Both are exposed **twice**: as LangChain tools bound to the agent's LLM, and as
MCP tools for external hosts. `fetch_demographics` is deliberately *not* a
tool — it's plumbing for the agent's `context_extraction` node (RAG
demographic hard-filtering), not an action the LLM should choose.

---

## 4. Flow 1 — Agent Save Path ("Save this note to the patient's chart")

The routing itself is the safety gate (guardrail G5, *no auto-save without
confirmation*): `tool_execution` is only reachable when `parse_input`
classified the request as an explicit `save` intent.

```
Physician: "Save this note to the patient's chart."
    │
    ▼
parse_input ──── intent="save" ────► tool_execution
                                          │
                                          │ 1. Build request context from state:
                                          │    user_input + patient_id + draft_note
                                          │    + suggested ICD codes
                                          │
                                          │ 2. get_tool_llm()
                                          │    = main LLM .bind_tools([save_note,
                                          │                            get_patient_history])
                                          │    invoked with TOOL_SYSTEM_PROMPT
                                          │
                                          ▼
                             LLM returns tool_calls
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │ no tool call              │ save_note(args)           │ EHR down
              ▼                           ▼                           ▼
   ToolResult ok=False;        save_note.invoke(args)        EhrApiError caught:
   LLM's explanation of        └► ehr_client.post_note()     ToolResult ok=False,
   what's missing                 └► POST /notes             "note was NOT saved",
   ("no draft note provided")     └► {"status":"saved",      error on errors channel
                                       "note_id":"N_ab12cd34"}
                                          │
                                          ▼
                             ToolResult ok=True, note_id="N_ab12cd34"
                                          │
                                          ▼
                             response_generation ──► "Note saved to patient
                                                     P001's chart (note ID N_ab12cd34)."
```

`TOOL_SYSTEM_PROMPT` (in `agent/prompts.py`) constrains the LLM: use only the
patient ID and note content given in the context, never invent either, and make
no tool call if a save is requested without a draft note.

### Demographics side-flow (also Task 11)

```
context_extraction (soap path)
    │ caller supplied patient_sex?  ──yes──► pass through (caller wins)
    │ no
    ▼
ehr_client.fetch_demographics(patient_id) ──► GET /patients/{id}
    │                                              │
    │ EhrApiError (EHR down)                       │ {"status":"found","patient":{age,sex}}
    ▼                                              ▼
patient_sex="unknown"                    age/sex flow into the RAG
(log warning; never blocks the note)     metadata hard-filter
```

---

## 5. Flow 2 — External MCP Client

An MCP host (Claude Desktop, Claude Code, an IDE) uses the same operations
without touching the agent:

```
MCP host (e.g. Claude Desktop)
    │  spawns process, JSON-RPC over stdin/stdout
    ▼
mcp/server.py (FastMCP "mednote-ehr")
    │
    ├── initialize            ◄── MCP handshake
    ├── tools/list            ──► [save_note, get_patient_history] + JSON schemas
    │                             (generated from type hints + docstrings)
    └── tools/call
         name="save_note"
         arguments={"patient_id": "P001", "note": "..."}
              │
              ▼
         ehr_client.post_note()  ──HTTP──►  FastAPI EHR (port 8100)
              │
              ▼
         result content: {"status": "saved", "note_id": "N_...", ...}
```

The FastAPI EHR server must be running for `tools/call` to succeed; if it
isn't, the `EhrApiError` surfaces as an MCP tool error rather than a crash.

---

## 6. Error Handling & Safety Summary

| Failure | Where caught | Behavior |
|---------|--------------|----------|
| EHR unreachable | `ehr_client` → `EhrApiError` | Agent: "the note was NOT saved… try again or save manually" on the errors channel; MCP: tool error |
| Unknown patient | EHR API (404 envelope) | `ToolResult ok=False` with the API's message |
| Missing/empty fields | EHR API (422 validation) | Rejected before any write |
| Save without a draft note | `TOOL_SYSTEM_PROMPT` + node | LLM makes no tool call; user told what's missing |
| Auto-save prevention (G5) | Graph routing | `tool_execution` reachable only via explicit `save` intent |

---

## 7. Test Coverage (20 tests)

| File | What it proves |
|------|----------------|
| `tests/test_ehr_api.py` | All endpoints, envelopes, validation, and seed↔dataset demographic alignment |
| `tests/test_tools.py` | LangChain tools round-trip through the real ASGI app; schemas exposed to the LLM; `EhrApiError` on dead port |
| `tests/test_mcp_server.py` | Both MCP tools registered with schemas; `call_tool` round-trip persists to the store |
| `tests/test_agent_graph.py` | Full graph save path → `note_id` in the reply; EHR-down degradation; no-tool-call path; demographics fetch + fallback |

All tool tests inject `TestClient(ehr_api.app)` through the `get_client()`
seam — real HTTP semantics, zero processes, temp-file stores.
