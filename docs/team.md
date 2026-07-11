# MedNote Scribe: Team & Stack

## Team & Roles

Solo build — all role areas below are owned by the 3 developers across the 4-week plan (`tasks.md` Task 1 role categories):

| Role Area | Owner | Covers |
|-----------|-------|--------|
| Prompt / RAG | Bibek Rauth | System prompts, ICD-10 ETL, embeddings, hybrid retrieval, re-ranking |
| Tools / MCP | Sandeep| Mock EHR API, `save_note`/`get_patient_history` tools, MCP server |
| Memory | Sandeep | SQLite visit-history store, cross-session recall |
| Guardrails / Caching | Kevin/Sandeep | Red-flag detection, diagnosis-assertion checks, RAG cache |
| Observability / UI | Kevin | Gradio UI, trace logging, dashboard, evals |

- [x] `docs/requirements.md` read and understood
- [x] `docs/tasks.md` (4-week task breakdown) read and understood
- [x] `docs/implementation_plan.md` (technical decisions, architecture, task-by-task build) read and understood

## Stack Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| Agent Framework | LangGraph (StateGraph) | Explicit control flow, independently testable nodes |
| LLM | Claude Sonnet (main) / Claude Haiku or Gemini Flash (fast tasks), via a custom LangChain wrapper | Best instruction-following for clinical text; wrapper keeps provider swap-friendly |
| Vector Store | Qdrant, embedded local mode (`qdrant-client`, no server/Docker) | Hybrid dense+sparse search with zero infra to install or run |
| Embeddings | SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`) | Domain-specific medical synonym mapping (UMLS-trained) |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L6-v2` | Precision re-ranking of top-K retrieval candidates |
| UI | Gradio (ChatInterface → Blocks) | Fast to build, shareable link, supports trace/dashboard tabs |
| Memory | SQLite | Zero-config, sufficient for per-patient visit history |
| Caching | In-process LRU dict (`RAGCache`) | No Redis dependency; sufficient for session-level speedup |
| Mock EHR | FastAPI + local JSON store | Realistic REST surface without a real EHR/PHI |
| Configuration | `config.yml` + `src/mednote/config.py` (pydantic) | Single source of truth for all tunable parameters; `.env` reserved for secrets only |
| Package Mgmt | `uv` + `pyproject.toml` | Fast installs, lockfile-based reproducibility |
| Python | 3.11+ | `TypedDict` features, `X \| Y` union syntax |

Full rationale for each decision lives in `docs/implementation_plan.md` → Technical Decisions table.
