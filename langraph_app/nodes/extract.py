"""
extract_node — use an LLM to extract structured ExtractionResult from report text.

Model selection:
  has_attachments=True  → ChatOpenAI("gpt-4o") with vision + OCR context
  has_attachments=False → ChatOpenAI("gpt-4.1-mini") text-only (cheaper)

Both use .with_structured_output(ExtractionResult, method="function_calling")
which constrains the LLM to return only valid Pydantic schema fields —
structural defense against prompt injection even if sanitization is bypassed.

Uses guardrail-sanitized text and OCR where available.

Structured data from specialized attachment nodes is injected as text context
BEFORE the LLM call, then the verified structured fields overwrite the LLM's
output after the call — preventing hallucination of attachment-derived fields.

Raw image_data_urls are only sent for "unknown" and "document" attachment types;
specialized types (screenshot/vehicle/id_document/person) are already processed.
"""
from __future__ import annotations

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langraph_app.config.settings import get_settings
from langraph_app.models.report import ExtractionResult

_SYSTEM_PROMPT = (
    "You are a strict JSON extraction engine for a law enforcement incident reporting system. "
    "Extract structured information from the incident report. "
    "Return ONLY the fields defined in the schema — no prose, no markdown, no extra keys. "
    "If a field cannot be determined from the report, use the default value."
)


_EXTRACT_VISION_TYPES = {"unknown", "document", ""}


def _get_llm(vision: bool) -> ChatOpenAI:
    s = get_settings()
    model = "gpt-4o" if vision else "gpt-4.1-mini"
    return ChatOpenAI(
        model=model,
        api_key=s.openai_api_key,
        max_tokens=1024,
    ).with_structured_output(ExtractionResult, method="function_calling")


def _append_ocr_context(content: list, ocr_texts: list[str]) -> None:
    combined = "\n\n".join(
        f"[OCR from attachment {i + 1}]:\n{t}"
        for i, t in enumerate(ocr_texts) if t.strip()
    )
    if combined:
        content.append({"type": "text", "text": combined})


def _append_chat_context(content: list, chat_transcript: list) -> None:
    lines = []
    for m in chat_transcript:
        ts = f" ({m.get('timestamp', '')})" if m.get("timestamp") else ""
        lines.append(
            f"[{m.get('platform', '?')}] {m.get('sender', '?')}: {m.get('message', '')}{ts}"
        )
    content.append({"type": "text",
                    "text": "[Chat transcript from screenshot]:\n" + "\n".join(lines)})


def _append_vehicle_context(content: list, vehicle_extractions: list) -> None:
    best = max(vehicle_extractions, key=lambda v: len(v.get("plate", "")))
    content.append({"type": "text", "text": f"[Vehicle extraction]:\n{json.dumps(best)}"})


def _append_id_context(content: list, id_document_extractions: list) -> None:
    for i, id_doc in enumerate(id_document_extractions):
        content.append({"type": "text", "text": f"[ID document {i + 1}]:\n{json.dumps(id_doc)}"})


def _append_person_context(content: list, person_descriptions: list) -> None:
    for i, p in enumerate(person_descriptions):
        label = "SUSPECT" if p.get("is_suspect") else "person"
        face = p.get("face_match_result", {})
        match_note = (
            f" [FACE MATCH: {face.get('name', '')} (sim={face.get('similarity', 0):.2f})]"
            if face.get("matched") else ""
        )
        content.append({"type": "text",
                        "text": f"[{label} description {i + 1}]{match_note}:\n{p.get('appearance', '')}"})


def _append_unhandled_images(content: list, data_urls: list, attachment_types: list) -> None:
    for idx, data_url in enumerate(data_urls):
        img_type = attachment_types[idx] if idx < len(attachment_types) else "unknown"
        if img_type in _EXTRACT_VISION_TYPES:
            content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}})


def _build_content(state: dict, attachment_types: list) -> list:
    """Assemble the full HumanMessage content list from all available context."""
    text: str = (
        state.get("guardrail_sanitized_text")
        or state.get("text_input")
        or state.get("free_text")
        or ""
    )
    content: list = [{"type": "text", "text": f"Incident report:\n{text}"}]

    ocr_texts: list[str] = state.get("guardrail_sanitized_ocr") or state.get("ocr_texts") or []
    if ocr_texts:
        _append_ocr_context(content, ocr_texts)

    visual_description: str = state.get("visual_description", "")
    if visual_description:
        content.append({"type": "text", "text": f"[Visual description]:\n{visual_description}"})

    chat_transcript: list = state.get("chat_transcript", [])
    if chat_transcript:
        _append_chat_context(content, chat_transcript)

    vehicle_extractions: list = state.get("vehicle_extractions", [])
    if vehicle_extractions:
        _append_vehicle_context(content, vehicle_extractions)

    id_document_extractions: list = state.get("id_document_extractions", [])
    if id_document_extractions:
        _append_id_context(content, id_document_extractions)

    person_descriptions: list = state.get("person_descriptions", [])
    if person_descriptions:
        _append_person_context(content, person_descriptions)

    if state.get("has_attachments"):
        data_urls: list[str] = state.get("image_data_urls", [])
        _append_unhandled_images(content, data_urls, attachment_types)

    return content


def _merge_structured_fields(result_dict: dict, state: dict, attachment_types: list) -> dict:
    """Overwrite LLM output with verified structured data from specialized nodes."""
    result_dict["attachment_types"] = attachment_types
    chat_transcript = state.get("chat_transcript", [])
    if chat_transcript:
        result_dict["chat_transcript"] = chat_transcript
    id_document_extractions = state.get("id_document_extractions", [])
    if id_document_extractions:
        result_dict["id_document"] = id_document_extractions[0]
    person_descriptions = state.get("person_descriptions", [])
    if person_descriptions:
        result_dict["person_descriptions"] = person_descriptions
    return result_dict


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    attachment_types: list = state.get("attachment_types", [])
    content = _build_content(state, attachment_types)
    llm = _get_llm(vision=state.get("has_attachments", False))
    messages = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=content)]

    try:
        result: ExtractionResult = llm.invoke(messages)
        result_dict = _merge_structured_fields(result.model_dump(), state, attachment_types)
        return {"extraction_result": result_dict}
    except Exception as exc:
        return {"error": f"extract_node: LLM extraction failed: {exc}"}
