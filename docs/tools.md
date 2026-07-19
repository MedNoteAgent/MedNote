# Tool Specifications (Task 10)

Formal specification of the MedNote EHR tools. Written **as-built** (2026-07-19,
`ft_mcp_build`): every envelope below is the implemented behavior, pinned by
tests in `tests/test_ehr_api.py`, `tests/test_tools.py`, and
`tests/test_mcp_server.py`. Architecture and flow diagrams live in
[`mcp_tools_notes.md`](mcp_tools_notes.md).

## Surfaces

Each tool is exposed twice, both delegating to the same implementation
(`src/mednote/tools/ehr_client.py`), so the surfaces cannot drift:

| Surface | Module | Consumer |
|---------|--------|----------|
| LangChain `@tool` | `tools/save_note.py`, `tools/get_history.py` | LangGraph agent (`tool_execution` node via `bind_tools`) |
| MCP (FastMCP, stdio) | `mcp/server.py` (server name `mednote-ehr`) | External MCP hosts (Claude Desktop / Code, Inspector) |

Failure contract: if the EHR API is unreachable or returns a non-JSON body,
`ehr_client` raises `EhrApiError`. Callers must degrade gracefully — the agent
reports *"the note was NOT saved"* on the errors channel; it never surfaces a
stack trace.

---

## Tool: `save_note`

Save a clinical note to a patient's EHR chart.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `patient_id` | string | Yes | Unique patient identifier (e.g., `"P001"`) — must exist in the EHR |
| `note` | string | Yes | The complete SOAP note text (non-empty) |
| `icd_codes` | list[string] | No | Suggested ICD-10 codes (e.g., `["G44.2"]`); defaults to `[]` |

**Backing endpoint:** `POST /notes`

**Success (200):**
```json
{
  "status": "saved",
  "note_id": "N_ab12cd34",
  "patient_id": "P001",
  "timestamp": "2026-07-19T02:56:55.186008+00:00"
}
```

**Errors:**
- Unknown patient → **404** `{"status": "error", "message": "Unknown patient ID 'X'."}`
- Missing/empty `patient_id` or `note` → **422** (pydantic validation, rejected before any write)

**Safety (guardrail G5 — no auto-save without confirmation):** the agent
reaches this tool only via an explicit `save` intent (graph routing is the
gate), and `TOOL_SYSTEM_PROMPT` forbids the LLM from inventing patient IDs or
note content — a save request without a provided note is declined, not
fabricated. The tool description carries the confirmation rule so every model
that binds it sees it.

---

## Tool: `get_patient_history`

Retrieve prior visit notes for a patient.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `patient_id` | string | Yes | Unique patient identifier |

**Backing endpoint:** `GET /patients/{patient_id}/history`

**Found (200):**
```json
{
  "status": "found",
  "patient_id": "P001",
  "visits": [
    {
      "note_id": "N_ab12cd34",
      "patient_id": "P001",
      "date": "2026-07-19",
      "note": "### Subjective\n...",
      "icd_codes": ["G44.2"],
      "created_at": "2026-07-19T02:56:55.186008+00:00"
    }
  ]
}
```

**No history (200):** `{"status": "no_history", "patient_id": "P001", "message": "No prior visits found"}`

**Error:** unknown patient → **404** `{"status": "error", "message": "..."}`

> Note: the graph's `history` **intent** is served by the SQLite memory store
> (Tasks 14–15, currently a stub) — not by this tool. The tool exists for
> LLM/MCP-initiated lookups; the two answer different questions today.

---

## Internal endpoint: patient demographics (not a tool)

`GET /patients/{patient_id}` → **200**
`{"status": "found", "patient": {"patient_id": "P005", "name": "Aarav Patel", "age": 4, "sex": "male"}}`
(unknown patient → **404** error envelope)

Used by the agent's `context_extraction` node to build the RAG demographic
hard-filter (Step 7.1). Deliberately **not** exposed as a tool: it is
app-controlled plumbing, not an action the LLM should choose. If the EHR is
down, the node degrades to `patient_sex="unknown"` rather than blocking the
note.

---

## Mock EHR server

- **App:** `src/mednote/tools/ehr_api.py` (FastAPI) — run
  `uv run uvicorn mednote.tools.ehr_api:app --port 8100`
- **Config:** host/port from `config.yml → ehr_api`; store path from
  `config.yml → paths.ehr_store_path`
- **Store:** `data/ehr_store.json` (gitignored, runtime-mutated), created on
  first write from `SEED_PATIENTS` — P001–P013 matching the Task 4 transcript
  demographics, plus the UI's `P-DEMO`. All data synthetic; no real PHI.
- **Production mapping:** see `mcp_tools_notes.md` — swap `ehr_client` for an
  authenticated FHIR client (`POST /DocumentReference`, `GET /Patient/{id}`,
  `GET /Encounter?patient={id}`); tool names, schemas, and agent code stay
  unchanged.
