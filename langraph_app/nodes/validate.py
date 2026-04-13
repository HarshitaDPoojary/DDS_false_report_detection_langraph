"""
validate_node — merge extraction result with defaults, hash SOC, extract quotes.

Takes extraction_result from extract_node and produces a final
validated_report dict ready for ingestion.

Imports legacy helpers:
  hash_soc()    → SHA-256 pseudonymous hash of subject-of-concern identity
  quick_quotes() → extract verbatim quoted speech from free_text
"""
from __future__ import annotations

from langraph_app.utils.text_utils import hash_soc, quick_quotes
from langraph_app.models.report import ExtractionResult


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    extraction_dict: dict = state.get("extraction_result", {})
    if not extraction_dict:
        return {"error": "validate_node: no extraction_result in state"}

    # Validate through Pydantic to enforce field types and defaults
    try:
        result = ExtractionResult(**extraction_dict)
    except Exception as exc:
        return {"error": f"validate_node: ExtractionResult validation failed: {exc}"}

    validated = result.model_dump()

    # ── SOC hash ──────────────────────────────────────────────────────────────
    # Hash the primary subject-of-concern (first named person + org, if present)
    named = result.who.named_persons or []
    org = result.who.target_org or ""
    soc_name = named[0] if named else ""
    soc_hash = hash_soc(soc_name, org) if soc_name or org else ""

    # ── Quotes ────────────────────────────────────────────────────────────────
    free_text: str = (
        state.get("guardrail_sanitized_text")
        or state.get("text_input")
        or state.get("free_text")
        or ""
    )
    quotes = quick_quotes(free_text)
    validated["quotes"] = quotes

    # ── screens_evidence ──────────────────────────────────────────────────────
    screens_evidence: bool = bool(state.get("image_data_urls") or state.get("ocr_texts"))

    # ── Merge context fields from state ───────────────────────────────────────
    validated.update({
        "report_id":               state.get("report_id", ""),
        "free_text":               free_text,
        "image_metadata":          state.get("image_metadata", []),
        "visual_description":      state.get("visual_description", ""),
        "screens_evidence":        screens_evidence,
        "soc_key":                 soc_hash,
        "reporter":                state.get("form_metadata", {}).get("reporter", {}),
        # Attachment classification + specialized extraction
        "attachment_types":        state.get("attachment_types", []),
        "chat_transcript":         state.get("chat_transcript", []),
        "vehicle_extractions":     state.get("vehicle_extractions", []),
        "id_document_extractions": state.get("id_document_extractions", []),
        "person_descriptions":     state.get("person_descriptions", []),
    })

    return {
        "validated_report": validated,
        "soc_hash": soc_hash,
        "screens_evidence": screens_evidence,
    }
