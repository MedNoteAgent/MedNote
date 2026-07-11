# MedNote

MedNote is a clinical documentation agent that turns doctor-patient transcripts into structured SOAP notes, suggests ICD-10 codes through a retrieval-augmented workflow, and can save notes through a mock EHR tool.

## Prerequisites

- Python 3.11+
- uv
- An API key for your preferred LLM provider (Anthropic, OpenAI, or Google)

## Installation

```bash
uv sync
```

## Development

```bash
uv sync --extra dev
```

## Run the app

```bash
uv run python -m mednote.ui.app
```
