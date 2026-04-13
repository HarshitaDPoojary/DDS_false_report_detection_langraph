"""
guardrails_node — deterministic pre-LLM security checks.

Runs as the FIRST node in IntakeGraph and AnalysisGraph.
Does NOT call any LLM.

Hard blocks (sets guardrail_hard_block=True, writes error, graph routes to END):
- Text input exceeds 50,000 characters
- More than 10 attachments
- Any attachment exceeds 20 MB
- File magic bytes don't match any allowed type / blocked extension

Flag-only (adds to guardrail_flags, processing continues):
- Prompt injection patterns in free_text / text_input
- Prompt injection patterns in OCR text (+ offending lines stripped)
- Prompt injection in EXIF metadata fields
- Text exceeds 20,000 characters
- report_id contains unsafe characters

Downstream nodes must:
1. Guard at entry: if state.get("guardrail_hard_block"): return {}
2. Use sanitized text: state.get("guardrail_sanitized_text") or state.get("free_text", "")
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from langraph_app.utils.sanitize import (
    TEXT_HARD_LIMIT_CHARS,
    MAX_ATTACHMENTS,
    check_text_size,
    check_image_size,
    validate_file_type,
    detect_injection,
    sanitize_for_llm,
    sanitize_ocr_texts,
    check_exif_injection,
    validate_report_id,
)


def _try_check_exif(image_bytes: bytes) -> list[dict]:
    """Extract and scan EXIF fields for injection. Gracefully fails if Pillow absent."""
    try:
        import io
        from PIL import Image, ExifTags
        img = Image.open(io.BytesIO(image_bytes))
        raw_exif = img.getexif() or {}
        exif_dict = {
            ExifTags.TAGS.get(tid, str(tid)): str(v)
            for tid, v in raw_exif.items()
        }
        return check_exif_injection(exif_dict)
    except Exception:
        return []


def _check_text(
    raw_text: str,
    text_field: str,
    flags: list[dict],
) -> tuple[bool, Optional[str]]:
    """Check text size and scan for injection. Returns (hard_block, block_reason)."""
    if not raw_text:
        return False, None

    is_hard, is_warn, char_count = check_text_size(raw_text)

    if is_hard:
        reason = (
            f"Input text exceeds hard limit "
            f"({char_count:,} chars > {TEXT_HARD_LIMIT_CHARS:,})"
        )
        flags.append({
            "check": "TEXT_SIZE_BLOCK",
            "field": text_field,
            "severity": "block",
            "detail": reason,
            "truncated_sample": raw_text[:80],
        })
        return True, reason

    if is_warn:
        flags.append({
            "check": "TEXT_SIZE_WARN",
            "field": text_field,
            "severity": "warn",
            "detail": f"Input text is large ({char_count:,} chars). May increase LLM cost.",
            "truncated_sample": raw_text[:80],
        })

    flags.extend(detect_injection(raw_text, field_name=text_field))
    return False, None


def _check_attachment(
    file_bytes: bytes,
    filename: str,
    idx: int,
    flags: list[dict],
) -> tuple[bool, Optional[str]]:
    """
    Validate a single attachment: size, file type, EXIF injection.
    Returns (hard_block, block_reason).
    """
    if check_image_size(file_bytes):
        size_mb = len(file_bytes) / (1024 * 1024)
        reason = f"Attachment '{filename}' exceeds 20 MB limit ({size_mb:.1f} MB)"
        flags.append({
            "check": "IMAGE_SIZE_BLOCK",
            "field": f"attachments[{idx}]",
            "severity": "block",
            "detail": reason,
            "truncated_sample": filename[:80],
        })
        return True, reason

    is_allowed, detected_mime = validate_file_type(filename, file_bytes)
    if not is_allowed:
        reason = (
            f"Attachment '{filename}' failed type validation "
            f"(detected: {detected_mime})"
        )
        flags.append({
            "check": "FILE_TYPE_BLOCK",
            "field": f"attachments[{idx}]",
            "severity": "block",
            "detail": reason,
            "truncated_sample": filename[:80],
        })
        return True, reason

    if detected_mime.startswith("image/"):
        exif_flags = _try_check_exif(file_bytes)
        for f in exif_flags:
            f["field"] = f"attachments[{idx}].{f.get('field', 'exif')}"
        flags.extend(exif_flags)

    return False, None


def _check_attachments(
    attachments: list,
    attachment_names: list,
    flags: list[dict],
) -> tuple[bool, Optional[str]]:
    """
    Validate all attachments. Returns (hard_block, first_block_reason).
    """
    if len(attachments) > MAX_ATTACHMENTS:
        reason = f"Too many attachments: {len(attachments)} (max {MAX_ATTACHMENTS})"
        flags.append({
            "check": "TOO_MANY_ATTACHMENTS",
            "field": "attachments",
            "severity": "block",
            "detail": reason,
            "truncated_sample": "",
        })
        return True, reason

    names = list(attachment_names) + ["unknown"] * len(attachments)
    first_reason: Optional[str] = None
    blocked = False

    for idx, file_bytes in enumerate(attachments):
        if not isinstance(file_bytes, (bytes, bytearray)):
            continue
        is_hard, reason = _check_attachment(file_bytes, names[idx], idx, flags)
        if is_hard:
            blocked = True
            if first_reason is None:
                first_reason = reason

    return blocked, first_reason


def run(state: dict) -> dict:
    """
    Guardrail node entry point.

    Works for both IntakeState (has text_input / attachments) and
    AnalysisState / IngestionState (has free_text).
    """
    flags: list[dict] = []
    hard_block = False
    block_reason: Optional[str] = None

    is_intake = "text_input" in state or "attachments" in state
    raw_text: str = state.get("text_input") or state.get("free_text") or ""
    text_field = "text_input" if is_intake else "free_text"

    # 1. report_id validation
    report_id: str = state.get("report_id", "")
    if report_id:
        _, id_flags = validate_report_id(report_id)
        flags.extend(id_flags)

    # 2 + 3. Text size check + injection scan
    text_blocked, text_reason = _check_text(raw_text, text_field, flags)
    if text_blocked:
        hard_block = True
        block_reason = text_reason

    # 4. Text sanitization
    sanitized_text = sanitize_for_llm(raw_text) if raw_text else ""

    # 5. Attachment checks
    attachments: list = state.get("attachments", [])
    attachment_names: list = state.get("attachment_names", [])
    att_blocked, att_reason = _check_attachments(attachments, attachment_names, flags)
    if att_blocked:
        hard_block = True
        block_reason = block_reason or att_reason

    # 6. OCR text sanitization (if already present in state)
    ocr_texts: list = state.get("ocr_texts", [])
    sanitized_ocr = ocr_texts
    if ocr_texts:
        sanitized_ocr, ocr_flags = sanitize_ocr_texts(ocr_texts)
        flags.extend(ocr_flags)

    # 7. Audit entry
    audit_entry = {
        "node": "guardrails_node",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hard_block": hard_block,
        "flag_count": len(flags),
        "flag_severities": {
            "block": sum(1 for f in flags if f.get("severity") == "block"),
            "warn":  sum(1 for f in flags if f.get("severity") == "warn"),
            "info":  sum(1 for f in flags if f.get("severity") == "info"),
        },
    }

    result: dict = {
        "guardrail_flags": flags,
        "guardrail_hard_block": hard_block,
        "guardrail_sanitized_text": sanitized_text,
        "guardrail_sanitized_ocr": sanitized_ocr,
        "audit_trail": [audit_entry],
    }

    if hard_block:
        result["error"] = block_reason or "Guardrail hard block triggered"

    return result
