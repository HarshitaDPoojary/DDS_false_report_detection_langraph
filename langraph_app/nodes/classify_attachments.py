"""
classify_attachments_node — classify each attachment by type before specialized processing.

Runs after check_attachments_node, before the parallel fan-out.

Classification labels (one per attachment, parallel to the attachments list):
  screenshot   — phone/desktop screen capture (chat, app, browser)
  vehicle      — car, truck, motorcycle or other road vehicle
  id_document  — driver's license, passport, state ID, or similar government ID
  person       — photograph of a human face or body
  document     — printed/handwritten document, form, or letter (non-ID)
  unknown      — none of the above, or classification failed

Two-phase classification per file:
  Phase A: filename heuristics (O(1), no model call)
  Phase B: vLLM single-token prompt (only when filename is ambiguous)

Sets boolean convenience flags:
  has_screenshot, has_vehicle, has_id_document, has_person

Does NOT write image_data_urls — that stays ocr_node's responsibility.
Calls to_data_url() internally only for the vLLM classification call.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from langraph_app.utils.vllm_client import get_vllm_client
from langraph_app.utils.image_utils import to_data_url

_VALID_LABELS = {"screenshot", "vehicle", "id_document", "person", "document", "unknown"}

_CLASSIFIER_PROMPT = (
    "Classify this image into exactly one category. "
    "Respond with ONLY one word — no punctuation, no explanation:\n"
    "screenshot   (a phone or desktop screen capture showing a chat, app, or website)\n"
    "vehicle      (a car, truck, motorcycle, or other road vehicle)\n"
    "id_document  (a driver's license, passport, state ID, or other government-issued ID)\n"
    "person       (a photograph of a human face or body)\n"
    "document     (a printed or handwritten document, form, or letter)\n"
    "other        (none of the above)\n"
    "Category:"
)

# Filename keyword hints — checked before calling vLLM
_FILENAME_HINTS: list[tuple[list[str], str]] = [
    (["screenshot", "screen", "_ss_", "-ss-", "snap", "capture", "chat"], "screenshot"),
    (["plate", "car", "vehicle", "auto", "truck", "suv", "sedan"], "vehicle"),
    (["license", "licence", "dl_", "_dl_", "passport", "id_card", "permit", "govid"], "id_document"),
    (["person", "face", "suspect", "victim", "mugshot", "photo_id"], "person"),
]


def _classify_by_filename(filename: str) -> str | None:
    """Return a label if the filename contains unambiguous keywords, else None."""
    name_lower = filename.lower()
    if name_lower.endswith(".pdf"):
        return "document"
    for keywords, label in _FILENAME_HINTS:
        if any(kw in name_lower for kw in keywords):
            return label
    return None


def _classify_by_vllm(file_bytes: bytes) -> str:
    """Call vLLM with a cheap single-token prompt to classify the image."""
    try:
        data_url = to_data_url(file_bytes)
        llm = get_vllm_client(max_tokens=10)
        msg = HumanMessage(content=[
            {"type": "text", "text": _CLASSIFIER_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
        ])
        response = llm.invoke([msg])
        label = response.content.strip().lower().split()[0].rstrip(".")
        return label if label in _VALID_LABELS else "unknown"
    except Exception:
        return "unknown"


def _classify_single(file_bytes: bytes, filename: str) -> str:
    """Classify one attachment. Phase A: filename heuristic. Phase B: vLLM."""
    label = _classify_by_filename(filename)
    if label is not None:
        return label
    if not isinstance(file_bytes, (bytes, bytearray)) or len(file_bytes) == 0:
        return "unknown"
    return _classify_by_vllm(bytes(file_bytes))


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}
    if not state.get("has_attachments"):
        return {}

    attachments: list = state.get("attachments", [])
    attachment_names: list = state.get("attachment_names", [])
    names = list(attachment_names) + ["unknown"] * max(0, len(attachments) - len(attachment_names))

    attachment_types: list[str] = []
    for idx, file_bytes in enumerate(attachments):
        label = _classify_single(file_bytes, names[idx])
        attachment_types.append(label)

    return {
        "attachment_types": attachment_types,
        "has_screenshot":   "screenshot"   in attachment_types,
        "has_vehicle":      "vehicle"       in attachment_types,
        "has_id_document":  "id_document"   in attachment_types,
        "has_person":       "person"        in attachment_types,
    }
