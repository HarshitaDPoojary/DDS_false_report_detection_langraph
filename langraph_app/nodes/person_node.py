"""
person_node — appearance description and optional face recognition for person photos.

Runs in parallel after classify_attachments_node, only when has_person=True.

Suspect context detection (either condition triggers face recognition attempt):
  - The report text contains the word "suspect" (case-insensitive)
  - form_metadata contains {"suspect_photo": True}

For each person image:
  1. vLLM generates a factual physical appearance description (clothing, hair,
     build, distinguishing features). No identity claims — description only.
  2. If suspect context AND face_recognition_enabled AND known_offender_db_path:
     InsightFace embeds the face and cosine-searches the known-offender DB.
     Result: {matched, offender_id, name, similarity} or {matched: False, reason}.

InsightFace is optional — the node works without it installed. Face recognition
only activates when ALL THREE conditions are met: enabled flag, DB path set,
AND the library is importable.

Returns: person_descriptions — list of {appearance, is_suspect, face_match_result}.
"""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage

from langraph_app.config.settings import get_settings
from langraph_app.utils.vllm_client import get_vllm_client
from langraph_app.utils.image_utils import to_data_url

_APPEARANCE_PROMPT = (
    "Describe the person visible in this photograph for a law enforcement incident report.\n"
    "Include: approximate age range, gender presentation, skin tone, hair color and style, "
    "clothing description (colors, type, logos if visible), build and height estimate if "
    "possible, any distinguishing features (tattoos, scars, glasses, facial hair, piercings).\n"
    "Be factual and objective. Do not name or identify the person.\n"
    "Return a single paragraph of plain text only."
)

_SUSPECT_PATTERN = re.compile(r"\bsuspect\b", re.IGNORECASE)


def _is_suspect_context(state: dict) -> bool:
    text = (
        state.get("guardrail_sanitized_text")
        or state.get("text_input")
        or state.get("free_text")
        or ""
    )
    if _SUSPECT_PATTERN.search(text):
        return True
    form_metadata = state.get("form_metadata") or {}
    return bool(form_metadata.get("suspect_photo"))


def _describe_person(file_bytes: bytes) -> str:
    """Call vLLM for a physical appearance description."""
    try:
        data_url = to_data_url(file_bytes)
        llm = get_vllm_client(max_tokens=512)
        msg = HumanMessage(content=[
            {"type": "text", "text": _APPEARANCE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ])
        response = llm.invoke([msg])
        return response.content.strip()[:1000]
    except Exception as exc:
        return f"person description failed: {exc}"


def _run_face_recognition(file_bytes: bytes) -> dict:
    """
    Embed the face with InsightFace and cosine-search the known-offender DB.

    Returns {matched, offender_id, name, similarity} on a hit, or
    {matched: False, reason} on miss / error / library not installed.
    """
    try:
        import json
        import pathlib

        import numpy as np

        s = get_settings()
        db_path = pathlib.Path(s.known_offender_db_path)
        if not db_path.exists():
            return {"matched": False, "reason": "offender_db_not_found"}

        # Decode image bytes to numpy array for InsightFace
        import cv2  # opencv-python-headless
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"matched": False, "reason": "image_decode_failed"}

        from insightface.app import FaceAnalysis  # type: ignore[import]

        face_app = FaceAnalysis(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        face_app.prepare(ctx_id=0, det_size=(640, 640))

        faces = face_app.get(img)
        if not faces:
            return {"matched": False, "reason": "no_face_detected"}

        embedding = np.array(faces[0].normed_embedding, dtype=np.float32)

        # DB format: [{offender_id, name, embedding: [512 floats]}]
        db: list[dict] = json.loads(db_path.read_text(encoding="utf-8"))

        best_sim = -1.0
        best_entry: dict | None = None
        for entry in db:
            db_emb = np.array(entry["embedding"], dtype=np.float32)
            sim = float(np.dot(embedding, db_emb))
            if sim > best_sim:
                best_sim, best_entry = sim, entry

        threshold = s.face_match_threshold
        if best_sim >= threshold and best_entry is not None:
            return {
                "matched":     True,
                "offender_id": str(best_entry.get("offender_id", "")),
                "name":        str(best_entry.get("name", "")),
                "similarity":  round(best_sim, 4),
            }
        return {"matched": False, "similarity": round(max(best_sim, 0.0), 4)}

    except ImportError:
        return {"matched": False, "reason": "insightface_not_installed"}
    except Exception as exc:
        return {"matched": False, "reason": f"face_recognition_error: {exc}"}


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}
    if not state.get("has_person"):
        return {}

    s = get_settings()
    is_suspect = _is_suspect_context(state)
    face_recog_available = (
        s.face_recognition_enabled
        and bool(s.known_offender_db_path)
    )

    attachments: list = state.get("attachments", [])
    attachment_types: list = state.get("attachment_types", [])

    person_descriptions: list[dict] = []
    for idx, file_bytes in enumerate(attachments):
        if idx < len(attachment_types) and attachment_types[idx] == "person":
            if not isinstance(file_bytes, (bytes, bytearray)):
                continue

            appearance = _describe_person(bytes(file_bytes))
            face_match: dict = {}

            if is_suspect and face_recog_available:
                face_match = _run_face_recognition(bytes(file_bytes))

            person_descriptions.append({
                "appearance":        appearance,
                "is_suspect":        is_suspect,
                "face_match_result": face_match,
            })

    return {"person_descriptions": person_descriptions}
