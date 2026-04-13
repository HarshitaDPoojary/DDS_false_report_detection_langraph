"""
id_document_node — extract structured identity fields from government-issued ID documents.

Runs in parallel after classify_attachments_node, only when has_id_document=True.

Uses vLLM (Qwen2-VL-7B or NIM fallback). Extracts:
  full_name, date_of_birth, address, id_number, issuer_state, expiry_date, document_type

The extracted name is surfaced to extract_node which adds it to
ExtractionResult.who.named_persons and ExtractionResult.id_document.

Returns: id_document_extractions — list of dicts (one per ID image).
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from langraph_app.utils.vllm_client import get_vllm_client
from langraph_app.utils.image_utils import to_data_url

_VALID_DOC_TYPES = {"drivers_license", "passport", "state_id", "unknown"}

_ID_PROMPT = (
    "Analyze this government-issued ID document for a law enforcement incident report.\n"
    "Extract the following fields and return ONLY a JSON object:\n"
    '{"full_name": "<full legal name as printed>", '
    '"date_of_birth": "<DOB as printed, e.g. 01/15/1985>", '
    '"address": "<full address as printed, or empty string>", '
    '"id_number": "<license, ID, or passport number>", '
    '"issuer_state": "<US state or country that issued this document>", '
    '"expiry_date": "<expiration date as printed, or empty string>", '
    '"document_type": "<drivers_license|passport|state_id|unknown>"}\n'
    "Rules:\n"
    "- Copy all text exactly as printed — do not correct spellings.\n"
    "- If a field is not visible or not present, use empty string.\n"
    "- Return ONLY the JSON object — no prose, no markdown."
)


def _normalize_id_doc(d: object) -> dict:
    if not isinstance(d, dict):
        return {
            "full_name": "", "date_of_birth": "", "address": "",
            "id_number": "", "issuer_state": "", "expiry_date": "",
            "document_type": "unknown",
        }
    doc_type = str(d.get("document_type", "unknown")).lower()
    if doc_type not in _VALID_DOC_TYPES:
        doc_type = "unknown"
    return {
        "full_name":     str(d.get("full_name",     ""))[:200],
        "date_of_birth": str(d.get("date_of_birth", ""))[:50],
        "address":       str(d.get("address",       ""))[:300],
        "id_number":     str(d.get("id_number",     ""))[:100],
        "issuer_state":  str(d.get("issuer_state",  ""))[:100],
        "expiry_date":   str(d.get("expiry_date",   ""))[:50],
        "document_type": doc_type,
    }


def _extract_id_document(file_bytes: bytes) -> dict:
    """Call vLLM to extract identity fields from one ID image."""
    try:
        data_url = to_data_url(file_bytes)
        llm = get_vllm_client(max_tokens=512)
        msg = HumanMessage(content=[
            {"type": "text", "text": _ID_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ])
        response = llm.invoke([msg])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return _normalize_id_doc(parsed)
    except Exception:
        return _normalize_id_doc({})


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}
    if not state.get("has_id_document"):
        return {}

    attachments: list = state.get("attachments", [])
    attachment_types: list = state.get("attachment_types", [])

    extractions: list[dict] = []
    for idx, file_bytes in enumerate(attachments):
        if idx < len(attachment_types) and attachment_types[idx] == "id_document":
            if isinstance(file_bytes, (bytes, bytearray)):
                extractions.append(_extract_id_document(bytes(file_bytes)))

    return {"id_document_extractions": extractions}
