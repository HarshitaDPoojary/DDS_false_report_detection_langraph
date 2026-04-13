"""
vehicle_node — extract plate, make/model, color, and damage from vehicle photos.

Runs in parallel after classify_attachments_node, only when has_vehicle=True.

Uses vLLM (Qwen2-VL-7B or NIM fallback). Qwen2-VL has no privacy restrictions
on license plate text, unlike GPT-4o.

Plate text is read verbatim — the model is explicitly instructed not to correct
or guess. If multiple vehicle images are submitted, all extractions are returned;
extract_node selects the best entry (longest plate text wins).

Returns: vehicle_extractions — list of {plate, make_model, color, damage_description}.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from langraph_app.utils.vllm_client import get_vllm_client
from langraph_app.utils.image_utils import to_data_url

_VEHICLE_PROMPT = (
    "Analyze this vehicle photograph for a law enforcement incident report.\n"
    "Return ONLY a JSON object with these exact keys:\n"
    '{"plate": "<license plate text exactly as printed — leave empty if not visible>", '
    '"make_model": "<vehicle make and model, e.g. Toyota Camry>", '
    '"color": "<primary color>", '
    '"damage_description": "<visible dents, scratches, broken parts, custom markings, ",'
    '"or empty string if none>"}\n'
    "Rules:\n"
    "- Copy plate characters exactly — do not correct, guess, or paraphrase.\n"
    "- If partial plate is visible, include what is legible.\n"
    "- Return ONLY the JSON object — no prose, no markdown."
)


def _normalize_vehicle(v: object) -> dict:
    if not isinstance(v, dict):
        return {"plate": "", "make_model": "", "color": "", "damage_description": ""}
    return {
        "plate":               str(v.get("plate",               ""))[:20],
        "make_model":          str(v.get("make_model",          ""))[:100],
        "color":               str(v.get("color",               ""))[:50],
        "damage_description":  str(v.get("damage_description",  ""))[:500],
    }


def _extract_vehicle(file_bytes: bytes) -> dict:
    """Call vLLM to extract vehicle details from one image."""
    try:
        data_url = to_data_url(file_bytes)
        llm = get_vllm_client(max_tokens=256)
        msg = HumanMessage(content=[
            {"type": "text", "text": _VEHICLE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ])
        response = llm.invoke([msg])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return _normalize_vehicle(parsed)
    except Exception as exc:
        return {
            "plate": "", "make_model": "", "color": "",
            "damage_description": f"vehicle_node error: {exc}",
        }


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}
    if not state.get("has_vehicle"):
        return {}

    attachments: list = state.get("attachments", [])
    attachment_types: list = state.get("attachment_types", [])

    extractions: list[dict] = []
    for idx, file_bytes in enumerate(attachments):
        if idx < len(attachment_types) and attachment_types[idx] == "vehicle":
            if isinstance(file_bytes, (bytes, bytearray)):
                extractions.append(_extract_vehicle(bytes(file_bytes)))

    return {"vehicle_extractions": extractions}
