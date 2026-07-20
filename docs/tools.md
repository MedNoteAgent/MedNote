# MedNote Scribe: Tool Specifications (Task 10)

Both tools front the mock EHR (Task 11): a small FastAPI REST surface backed
by a local JSON file (`data/ehr_store.json`, path from `config.yml` ->
`paths.ehr_store_path`). No real patient data / PHI — all inputs are
synthetic (`requirements.md` §4).

Agreed by the team as the contract both the LangChain tool wrappers
(`src/mednote/tools/`) and the MCP server (`src/mednote/mcp/server.py`,
Task 13) implement identically.

---

## `save_note`

Saves a clinical note to a patient's EHR chart. Corresponds to
`requirements.md` §3 Q3 ("Save this note to the patient's chart").

### Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `patient_id` | string | Yes | Unique patient identifier (e.g. `"P001"`) |
| `note` | string | Yes | The complete SOAP note text |
| `icd_codes` | list[string] | No | Suggested ICD-10 codes (e.g. `["G44.2"]`); defaults to `[]` |

### Outputs

**Success:**
```json
{
  "status": "saved",
  "note_id": "N_1a2b3c4d",
  "patient_id": "P001",
  "timestamp": "2026-07-20T14:32:00+00:00"
}
```

**Error — missing `patient_id`:**
```json
{"status": "error", "message": "Patient ID is required"}
```

**Error — missing `note`:**
```json
{"status": "error", "message": "Note text is required"}
```

### Notes
- `note_id` is generated server-side (`N_` + 8 hex chars); never supplied by the caller.
- Per `requirements.md` §5 guardrail 5, calling this tool at all is the explicit
  physician-confirmation step — the agent must never call it automatically.
- Writing a note also appends a visit summary to the memory store (Task 14)
  so `get_patient_history` / the agent's memory recall (Task 15) can surface
  it in a later session.

---

## `get_patient_history`

Retrieves a patient's prior visit notes from the EHR. Corresponds to
`requirements.md` §3 Q4 ("What did I note for this patient's last visit?").

### Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `patient_id` | string | Yes | Unique patient identifier |

### Outputs

**Found (one or more prior visits, newest first):**
```json
{
  "status": "found",
  "patient_id": "P001",
  "visits": [
    {
      "note_id": "N_1a2b3c4d",
      "patient_id": "P001",
      "note": "### Subjective\n...",
      "icd_codes": ["G44.1", "R51"],
      "timestamp": "2026-07-20T14:32:00+00:00"
    }
  ]
}
```

**No history (new patient / never saved a note):**
```json
{"status": "no_history", "patient_id": "P001", "message": "No prior visits found"}
```

**Error — missing `patient_id`:**
```json
{"status": "error", "message": "Patient ID is required"}
```

---

## Implementation notes (as built, Tasks 11-13)

- **Core logic lives in `EHRStore`** (`src/mednote/tools/ehr_api.py`): a
  thread-safe, JSON-file-backed store with `save_note()` / `get_history()`
  methods. Both the FastAPI endpoints and the LangChain `@tool` wrappers call
  the same store instance (`get_ehr_store()`), so there is exactly one
  implementation of the save/lookup contract above.
- **The FastAPI app is a fully independent REST server** (`POST /notes`,
  `GET /patients/{patient_id}/history`), runnable on its own
  (`uv run uvicorn mednote.tools.ehr_api:app --port 8100`) — this is the
  "realistic REST surface" `requirements.md` §4 asks for. The tool wrappers
  and the MCP server call `EHRStore` **in-process** rather than over HTTP:
  both run in the same Python process for this demo, so a network hop would
  only add latency and a port-management burden without changing the
  contract. Deviation from the implementation-plan's `httpx.post(...)`
  sketch, documented here for the same reason `docs/retrieval_process_notes.md`
  documents its own runtime deviations.
- **MCP** (`src/mednote/mcp/server.py`) exposes both tools via `FastMCP`
  (the `mcp` SDK's high-level server), with identical signatures and
  docstrings, so a general-purpose MCP client sees the same contract as the
  LangChain agent.
