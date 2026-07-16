"""Tag codes with demographic restrictions for hard filtering (Step 5.4).

A 45-year-old man must never be offered pregnancy codes: the retriever
filters on these tags BEFORE scoring, so demographically-invalid codes never
enter the candidate list.
"""

from __future__ import annotations

from dataclasses import replace

from mednote.rag.etl.parser import ICD10Code

SEX_RESTRICTIONS: dict[str, str] = {
    "O": "female",   # Chapter 15: Pregnancy, childbirth and the puerperium (O00-O9A)
    "N40": "male",   # Prostate disorders
    "N41": "male",
    "N42": "male",
}

AGE_RESTRICTIONS: dict[str, int] = {
    "P": 28,         # Chapter 16: Perinatal conditions (P00-P96) -> newborns (max_age_days)
}


def _sex_for(code: str) -> list[str]:
    for prefix, sex in SEX_RESTRICTIONS.items():
        if code.startswith(prefix):
            return [sex]
    return []


def _max_age_for(code: str) -> int | None:
    for prefix, days in AGE_RESTRICTIONS.items():
        if code.startswith(prefix):
            return days
    return None


def apply_metadata_tags(codes: list[ICD10Code]) -> list[ICD10Code]:
    """Return NEW code objects carrying sex/age tags; inputs are not mutated.

    Codes with no applicable restriction are passed through unchanged.
    """
    tagged: list[ICD10Code] = []
    for code in codes:
        target_sex = _sex_for(code.code)
        max_age_days = _max_age_for(code.code)
        if target_sex or max_age_days is not None:
            tagged.append(replace(code, target_sex=target_sex, max_age_days=max_age_days))
        else:
            tagged.append(code)
    return tagged
