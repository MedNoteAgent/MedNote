"""HTTP client for the mock EHR API — the single integration point.

Every surface that talks to the EHR (LangChain tools, the MCP server, agent
nodes) delegates to these functions, so behavior can never drift between
surfaces. Base URL comes from config.yml -> ehr_api.

Connection/protocol failures raise :class:`EhrApiError`; callers turn that
into a graceful "EHR unavailable" degradation, never a stack trace.
"""

from __future__ import annotations

import httpx

from mednote.config import get_config

_TIMEOUT_SECONDS = 10.0


class EhrApiError(RuntimeError):
    """The EHR API could not be reached or returned an unreadable response."""


def get_client() -> httpx.Client:
    """New client for one request cycle. Seam for tests (TestClient injection)."""
    ehr = get_config().ehr_api
    return httpx.Client(
        base_url=f"http://{ehr.host}:{ehr.port}", timeout=_TIMEOUT_SECONDS
    )


def _request(
    method: str, path: str, client: httpx.Client | None = None, **kwargs
) -> dict:
    try:
        if client is not None:
            response = client.request(method, path, **kwargs)
        else:
            with get_client() as owned:
                response = owned.request(method, path, **kwargs)
        return response.json()
    except httpx.HTTPError as exc:
        raise EhrApiError(f"EHR API unreachable: {exc}") from exc
    except ValueError as exc:  # non-JSON body
        raise EhrApiError(f"EHR API returned a non-JSON response: {exc}") from exc


def post_note(
    patient_id: str,
    note: str,
    icd_codes: list[str] | None = None,
    client: httpx.Client | None = None,
) -> dict:
    """POST /notes — returns the API envelope (status saved | error)."""
    payload = {"patient_id": patient_id, "note": note, "icd_codes": icd_codes or []}
    return _request("POST", "/notes", client=client, json=payload)


def fetch_history(patient_id: str, client: httpx.Client | None = None) -> dict:
    """GET /patients/{id}/history — envelope (found | no_history | error)."""
    return _request("GET", f"/patients/{patient_id}/history", client=client)


def fetch_demographics(patient_id: str, client: httpx.Client | None = None) -> dict:
    """GET /patients/{id} — envelope (found | error) with age/sex payload."""
    return _request("GET", f"/patients/{patient_id}", client=client)
