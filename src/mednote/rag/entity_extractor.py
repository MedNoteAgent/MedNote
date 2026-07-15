"""Entity extraction & query rewriting via a fast LLM (Step 7.2).

Critical (docs/implementation_plan.md Task 7): do NOT pass the whole
transcript to the vector DB. Transcripts are full of noise — greetings,
scheduling, tangents. The SOAP Assessment section is the clinically relevant
distillation, and a fast LLM turns its colloquialisms ("ear infection in both
ears") into the formal terminology the index actually contains ("Acute
bilateral otitis media"). This is also what bridges the "heart attack" gap:
the ICD-10 Index has no card for the colloquialism, but it has one for the
normalized term.
"""

from __future__ import annotations

import json

ENTITY_EXTRACTION_PROMPT = """Extract the primary clinical conditions from this assessment.
Translate colloquial terms into standard clinical terminology.
Return as a JSON array of strings.

Example input: "Looks like the kid has an ear infection in both ears, and the mom's blood pressure is running high"
Example output: ["Acute bilateral otitis media", "Essential hypertension"]

Assessment:
{assessment_text}
"""


def _content_text(content) -> str:
    """Flatten a LangChain message content to text.

    Providers differ: some return a plain string, others (e.g. Gemini via
    langchain-google-genai) return a list of content blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get("type") == "text")
        ]
        return "".join(parts)
    raise ValueError(f"Unexpected LLM content type: {type(content).__name__}")


def _parse_entities(content) -> list[str]:
    """Parse the LLM reply into a clean list of entity strings.

    Raises:
        ValueError: if the reply is not a JSON array of strings.
    """
    text = _content_text(content).strip()
    if text.startswith("```"):
        # Strip a markdown fence (```json ... ```) some models insist on.
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Entity extraction reply is not valid JSON: {content!r}") from exc

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Entity extraction reply is not a JSON array of strings: {content!r}")

    return [item.strip() for item in data if item.strip()]


class EntityExtractor:
    """Rewrites a SOAP assessment into targeted, formal retrieval queries."""

    def __init__(self, llm=None):
        if llm is None:
            from mednote.llm.wrapper import get_fast_llm

            llm = get_fast_llm()
        self._llm = llm

    def extract(self, assessment_text: str) -> list[str]:
        """Extract normalized clinical entities from the assessment section.

        Raises:
            ValueError: if the assessment is blank or the LLM reply cannot be
                parsed — the caller (RAGPipeline) decides the fallback.
        """
        if not assessment_text or not assessment_text.strip():
            raise ValueError("Assessment text is empty")

        reply = self._llm.invoke(
            ENTITY_EXTRACTION_PROMPT.format(assessment_text=assessment_text.strip())
        )
        return _parse_entities(reply.content)
