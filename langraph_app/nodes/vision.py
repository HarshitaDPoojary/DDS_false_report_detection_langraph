"""
vision_node — general scene description for unknown/document image attachments.

Runs in parallel with ocr_node and image_metadata_node.
Only runs when has_attachments=True.

Uses vLLM (Qwen2-VL-7B or NVIDIA NIM fallback) instead of GPT-4o.

Skips images already handled by specialized nodes (screenshot_node, vehicle_node,
id_document_node, person_node) — those types produce structured output that is
richer than a prose description. vision_node only describes "unknown", "document",
and unclassified images.

Returns visual_description: a single prose string for use by extract_node.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from langraph_app.utils.vllm_client import get_vllm_client

_VISION_PROMPT = (
    "Describe what you see in this image for a law enforcement incident report system. "
    "Include: number of people (count, clothing, any visible weapons), vehicles (type, color, "
    "plate numbers if visible), location cues (street signs, landmarks, building type), "
    "actions taking place, any visible text or writing, and overall scene context. "
    "Be factual and concise. If nothing relevant is visible, say so."
)

# These types are handled by specialized nodes — skip them here
_SKIP_TYPES = {"screenshot", "vehicle", "id_document", "person"}


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    data_urls: list[str] = state.get("image_data_urls", [])
    if not data_urls:
        return {"visual_description": ""}

    attachment_types: list = state.get("attachment_types", [])

    # Only describe images not already handled by a specialized node
    filtered: list[tuple[int, str]] = [
        (idx, url) for idx, url in enumerate(data_urls)
        if idx >= len(attachment_types) or attachment_types[idx] not in _SKIP_TYPES
    ]

    if not filtered:
        return {"visual_description": ""}

    try:
        llm = get_vllm_client(max_tokens=512)
    except RuntimeError:
        return {"visual_description": ""}

    descriptions: list[str] = []
    for idx, data_url in filtered:
        try:
            msg = HumanMessage(content=[
                {"type": "text", "text": _VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url, "detail": "auto"}},
            ])
            response = llm.invoke([msg])
            descriptions.append(f"[Image {idx + 1}]: {response.content}")
        except Exception as exc:
            descriptions.append(f"[Image {idx + 1}]: vision analysis failed ({exc})")

    return {"visual_description": "\n\n".join(descriptions)}
