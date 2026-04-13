"""
screenshot_node — extract structured chat transcript from phone/desktop screenshots.

Runs in parallel after classify_attachments_node, only when has_screenshot=True.

Uses vLLM (Qwen2-VL-7B or NIM fallback) with a structured JSON prompt to extract
every visible message: sender, message text, timestamp, and platform.

Fallback: if vLLM is unavailable, runs pytesseract and wraps the raw text as a
single-entry transcript so downstream nodes still have something to work with.

Returns: chat_transcript — list of {sender, message, timestamp, platform} dicts.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from langraph_app.utils.vllm_client import get_vllm_client
from langraph_app.utils.image_utils import ocr_image_bytes, to_data_url

_TRANSCRIPT_PROMPT = (
    "Analyze this screenshot of a messaging conversation or communication app.\n"
    "Extract every visible message and return a JSON array. Each element must be:\n"
    '{"sender": "<name, phone number, or Unknown>", '
    '"message": "<exact message text>", '
    '"timestamp": "<date/time as shown, or empty string>", '
    '"platform": "<WhatsApp|SMS|iMessage|Signal|Telegram|Email|Facebook|Instagram|other>"}\n'
    "Rules:\n"
    "- Preserve the original order of messages (top to bottom).\n"
    "- Copy message text verbatim — do not paraphrase or summarize.\n"
    "- If this is not a chat screenshot, return [].\n"
    "Return ONLY the JSON array — no prose, no markdown code fences."
)

_MAX_SENDER_LEN = 200
_MAX_MESSAGE_LEN = 2000
_MAX_TIMESTAMP_LEN = 100
_MAX_PLATFORM_LEN = 50


def _normalize_message(m: object) -> dict | None:
    if not isinstance(m, dict):
        return None
    return {
        "sender":    str(m.get("sender",    "Unknown"))[:_MAX_SENDER_LEN],
        "message":   str(m.get("message",   ""))[:_MAX_MESSAGE_LEN],
        "timestamp": str(m.get("timestamp", ""))[:_MAX_TIMESTAMP_LEN],
        "platform":  str(m.get("platform",  "unknown"))[:_MAX_PLATFORM_LEN],
    }


def _extract_screenshot(file_bytes: bytes) -> list[dict]:
    """Call vLLM to extract a structured transcript from one screenshot."""
    try:
        data_url = to_data_url(file_bytes)
        llm = get_vllm_client(max_tokens=2048)
        msg = HumanMessage(content=[
            {"type": "text", "text": _TRANSCRIPT_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ])
        response = llm.invoke([msg])
        raw = response.content.strip()
        # Strip accidental markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        return [n for m in parsed if (n := _normalize_message(m)) is not None]
    except Exception:
        # Fallback: pytesseract raw OCR wrapped as a single entry
        try:
            raw_text = ocr_image_bytes(file_bytes)
            if raw_text.strip():
                return [{"sender": "unknown", "message": raw_text.strip(),
                         "timestamp": "", "platform": "unknown"}]
        except Exception:
            pass
        return []


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}
    if not state.get("has_screenshot"):
        return {}

    attachments: list = state.get("attachments", [])
    attachment_types: list = state.get("attachment_types", [])

    all_messages: list[dict] = []
    for idx, file_bytes in enumerate(attachments):
        if idx < len(attachment_types) and attachment_types[idx] == "screenshot":
            if isinstance(file_bytes, (bytes, bytearray)):
                messages = _extract_screenshot(bytes(file_bytes))
                all_messages.extend(messages)

    return {"chat_transcript": all_messages}
