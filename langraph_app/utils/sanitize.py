"""
Input sanitization utilities for the guardrails layer.

Pure functions — no LLM calls, no side effects, no external dependencies
beyond the stdlib `re` module and Pillow (optional, gracefully absent).

All guardrail checks are deterministic regex/byte-level inspection.
Using an LLM to check for prompt injection would be self-defeating.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Size limits
# ---------------------------------------------------------------------------
TEXT_HARD_LIMIT_CHARS = 50_000       # hard block above this
TEXT_WARN_LIMIT_CHARS = 20_000       # flag-only above this
OCR_MAX_CHARS = 10_000               # tighter limit for OCR text (noisier source)
IMAGE_HARD_LIMIT_BYTES = 20 * 1024 * 1024  # 20 MB per image
MAX_ATTACHMENTS = 10


# ---------------------------------------------------------------------------
# Allowed file types (magic byte signatures)
# ---------------------------------------------------------------------------
_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff",       "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n",  "image/png"),
    (b"GIF87a",             "image/gif"),
    (b"GIF89a",             "image/gif"),
    (b"RIFF",               "image/webp"),   # also check bytes[8:12] == b"WEBP"
    (b"BM",                 "image/bmp"),
    (b"II*\x00",            "image/tiff"),
    (b"MM\x00*",            "image/tiff"),
]

_PDF_MAGIC = b"%PDF-"

# These extensions are never accepted regardless of magic bytes
_BLOCKED_EXTENSIONS = {
    ".exe", ".dll", ".bat", ".cmd", ".sh", ".ps1", ".msi", ".scr",
    ".vbs", ".js", ".jar", ".app", ".dmg",
    ".svg",   # SVG can embed <script> tags and data: URI payloads
    ".html", ".htm", ".xml",
    ".php", ".py", ".rb",
}

# EXIF text fields that may carry injected content
_EXIF_INJECTION_FIELDS = {
    "ImageDescription", "UserComment", "Copyright", "Artist",
    "Make", "Model", "Software", "DocumentName", "XPComment",
    "XPAuthor", "XPTitle", "XPSubject", "XPKeywords",
}


# ---------------------------------------------------------------------------
# Prompt injection detection patterns
# ---------------------------------------------------------------------------

# Tier 1 — High-confidence (severity="warn")
# These match the grammatical structure of model-manipulation attempts.
# Legitimate emergency reports do not contain model-control directives.
_INJECTION_PATTERNS_HIGH: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
        re.IGNORECASE
    ), "classic_ignore_instruction"),

    (re.compile(
        r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
        re.IGNORECASE
    ), "disregard_instruction"),

    (re.compile(
        r"you\s+are\s+now\s+(a|an|the)\s+\w+",
        re.IGNORECASE
    ), "persona_override"),

    (re.compile(
        r"(act|behave|respond|pretend|roleplay)\s+as\s+(if\s+you\s+are\s+|a\s+|an\s+)?"
        r"(?:an?\s+)?(unrestricted|jailbroken|uncensored|unfiltered|DAN|evil|malicious)",
        re.IGNORECASE
    ), "jailbreak_persona"),

    (re.compile(
        r"(system\s+prompt|your\s+instructions?|your\s+training)\s*:?\s*"
        r"(is|are|was|were|will\s+be)?\s*(now|henceforth|from\s+now\s+on)",
        re.IGNORECASE
    ), "system_prompt_override"),

    (re.compile(
        r"\[SYSTEM\]|\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>|<\|system\|>",
        re.IGNORECASE
    ), "llm_control_token"),

    (re.compile(
        r"(output|print|return|reveal|show|display|repeat)\s+(your\s+)?"
        r"(system\s+prompt|initial\s+instructions?|base\s+prompt|hidden\s+instructions?)",
        re.IGNORECASE
    ), "system_prompt_exfil"),

    (re.compile(
        r"(do\s+not\s+follow|stop\s+following|bypass|override)\s+(your\s+)?"
        r"(guidelines?|rules?|restrictions?|safety|filters?|constraints?)",
        re.IGNORECASE
    ), "guideline_bypass"),

    (re.compile(
        r"-{3,}\s*(new\s+instructions?|updated\s+prompt|override)\s*-{3,}",
        re.IGNORECASE
    ), "delimiter_injection"),

    (re.compile(
        r"(as\s+(a|an)\s+)?(language\s+model|LLM|AI\s+assistant|GPT|Claude|chatbot)"
        r"[,\s]+(you\s+(must|should|will|shall|need\s+to))",
        re.IGNORECASE
    ), "ai_direct_address"),
]

# Tier 2 — Suspicious but ambiguous (severity="info")
# May appear in legitimate reports (e.g., a report about a phishing attack)
_INJECTION_PATTERNS_LOW: list[tuple[re.Pattern, str]] = [
    (re.compile(r"<script[\s>]", re.IGNORECASE), "script_tag"),
    (re.compile(r"javascript\s*:", re.IGNORECASE), "javascript_protocol"),
    (re.compile(r"data\s*:\s*text/(html|javascript)", re.IGNORECASE), "data_uri_html"),
    (re.compile(r"\{\{.*?\}\}|\{%.*?%\}", re.DOTALL), "template_injection"),
    (re.compile(r"<!--.*?-->", re.DOTALL), "html_comment"),
]

# LLM control token strip pattern (used in sanitize_for_llm)
_CONTROL_TOKENS = re.compile(
    r"<\|im_start\|>|<\|im_end\|>|<\|system\|>|<\|user\|>|<\|assistant\|>"
    r"|\[INST\]|\[/INST\]|\[SYSTEM\]|\[/SYSTEM\]"
    r"|<<SYS>>|<</SYS>>",
    re.IGNORECASE
)

# Safe report_id characters
_REPORT_ID_SAFE = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_text_size(text: str) -> tuple[bool, bool, int]:
    """
    Returns (is_hard_block, is_warn, char_count).
    Hard block at TEXT_HARD_LIMIT_CHARS; warn at TEXT_WARN_LIMIT_CHARS.
    """
    n = len(text)
    return n > TEXT_HARD_LIMIT_CHARS, n > TEXT_WARN_LIMIT_CHARS, n


def check_image_size(image_bytes: bytes) -> bool:
    """Returns True if the image exceeds the hard size limit."""
    return len(image_bytes) > IMAGE_HARD_LIMIT_BYTES


def validate_file_type(filename: str, file_bytes: bytes) -> tuple[bool, str]:
    """
    Returns (is_allowed, detected_mime).

    Hard blocks:
    - Extension in _BLOCKED_EXTENSIONS
    - Magic bytes do not match any known-safe image or PDF type
    """
    name_lower = filename.lower()
    ext = ("." + name_lower.rsplit(".", 1)[-1]) if "." in name_lower else ""

    if ext in _BLOCKED_EXTENSIONS:
        return False, "blocked_extension"

    if not file_bytes:
        return False, "empty_file"

    header = file_bytes[:16]

    if header.startswith(_PDF_MAGIC):
        return True, "application/pdf"

    for magic, mime in _IMAGE_MAGIC:
        if header.startswith(magic):
            if mime == "image/webp":
                if len(file_bytes) >= 12 and file_bytes[8:12] == b"WEBP":
                    return True, mime
                continue
            return True, mime

    return False, "unknown_binary"


def detect_injection(text: str, field_name: str = "text") -> list[dict]:
    """
    Run all injection pattern tiers against text.
    Returns list of GuardrailFlag dicts (may be empty).
    Never raises — errors are caught and returned as info flags.
    """
    flags: list[dict] = []
    try:
        for pattern, check_name in _INJECTION_PATTERNS_HIGH:
            m = pattern.search(text)
            if m:
                start = max(0, m.start() - 20)
                sample = text[start:m.end() + 20][:80]
                flags.append({
                    "check": f"INJECTION_{check_name.upper()}",
                    "field": field_name,
                    "severity": "warn",
                    "detail": f"High-confidence injection pattern '{check_name}' matched",
                    "truncated_sample": sample,
                })

        for pattern, check_name in _INJECTION_PATTERNS_LOW:
            m = pattern.search(text)
            if m:
                start = max(0, m.start() - 20)
                sample = text[start:m.end() + 20][:80]
                flags.append({
                    "check": f"SUSPICIOUS_{check_name.upper()}",
                    "field": field_name,
                    "severity": "info",
                    "detail": f"Suspicious pattern '{check_name}' — may be legitimate",
                    "truncated_sample": sample,
                })
    except Exception as exc:
        flags.append({
            "check": "INJECTION_SCAN_ERROR",
            "field": field_name,
            "severity": "info",
            "detail": f"Pattern scan failed: {type(exc).__name__}",
            "truncated_sample": "",
        })
    return flags


def sanitize_for_llm(text: str, max_chars: int = TEXT_HARD_LIMIT_CHARS) -> str:
    """
    Sanitize text for safe inclusion in LLM prompts.

    What is STRIPPED:
    - LLM control tokens (<|im_start|>, [INST], etc.)
    - HTML/XML tags (stripped, not escaped — tags serve no purpose in LLM prompts)
    - Null bytes and C0 control chars (except tab \\t, newline \\n, carriage return \\r)
    - Unicode bidirectional override characters (U+202A–U+202E, U+2066–U+2069)
      used in Trojan Source / bidirectional text attacks
    - Excessive repeated characters (capped to prevent embedding manipulation)

    What is PRESERVED:
    - All natural language including non-ASCII and non-English text
    - Quoted speech (single/double quotes kept)
    - Incident vocabulary (weapons, threats, violence — all valid content)
    - Newlines and whitespace
    """
    if not text:
        return ""

    # Step 1: Truncate first (fast path)
    text = text[:max_chars]

    # Step 2: Null bytes and C0 control chars (keep \t \n \r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Step 3: Unicode bidirectional override and zero-width chars
    text = re.sub(r"[\u202a-\u202e\u2066-\u2069\u200b-\u200f\ufeff]", "", text)

    # Step 4: LLM control tokens
    text = _CONTROL_TOKENS.sub("", text)

    # Step 5: HTML/XML tags
    # Rationale: a report about "he had a <knife>" loses angle brackets but keeps
    # the word. A report about a phishing <script> attack keeps the word "script".
    text = re.sub(r"<[^>]{0,500}>", "", text)

    # Step 6: Collapse excessive character repetition
    text = re.sub(r"([a-zA-Z])\1{4,}", r"\1\1\1\1", text)
    text = re.sub(r"([^a-zA-Z0-9\s])\1{3,}", r"\1\1\1", text)

    # Step 7: Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text


def sanitize_ocr_texts(ocr_texts: list[str]) -> tuple[list[str], list[dict]]:
    """
    Sanitize all OCR text strings.
    Returns (sanitized_list, all_flags).

    For OCR text, entire lines containing injection patterns are removed
    (not just individual characters). OCR output is line-independent —
    each line is a visually extracted segment, so removing one line does
    not corrupt surrounding content.
    """
    sanitized: list[str] = []
    all_flags: list[dict] = []

    for idx, ocr_text in enumerate(ocr_texts):
        field_name = f"ocr_texts[{idx}]"
        flags = detect_injection(ocr_text, field_name=field_name)
        all_flags.extend(flags)

        # If high-confidence injection found, strip the offending lines
        if any(f["severity"] == "warn" for f in flags if f["field"] == field_name):
            ocr_text = _strip_injection_lines(ocr_text)

        sanitized.append(sanitize_for_llm(ocr_text, max_chars=OCR_MAX_CHARS))

    return sanitized, all_flags


def _strip_injection_lines(text: str) -> str:
    """
    Remove individual lines from OCR text that contain high-confidence injection patterns.
    Replaced with a visible placeholder so the removal is auditable.
    """
    lines = text.splitlines()
    clean_lines: list[str] = []
    for line in lines:
        line_flags = detect_injection(line, field_name="_line")
        if any(f["severity"] == "warn" for f in line_flags):
            clean_lines.append("[GUARDRAIL: LINE REMOVED - INJECTION PATTERN]")
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines)


def check_exif_injection(exif_dict: dict) -> list[dict]:
    """
    Scan string-valued EXIF fields for injection patterns.
    exif_dict should be a flat dict mapping field name → string value.
    """
    flags: list[dict] = []
    for field_name in _EXIF_INJECTION_FIELDS:
        value = exif_dict.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        sub_flags = detect_injection(value, field_name=f"exif.{field_name}")
        flags.extend(sub_flags)
        # Also catch script-like content that might not match the regex patterns
        lower = value.lower()
        if any(kw in lower for kw in ("<script", "javascript:", "onerror=", "onload=")):
            flags.append({
                "check": "EXIF_SCRIPT_CONTENT",
                "field": f"exif.{field_name}",
                "severity": "warn",
                "detail": "EXIF field contains script-like content",
                "truncated_sample": value[:80],
            })
    return flags


def validate_report_id(report_id: str) -> tuple[bool, list[dict]]:
    """
    Validates report_id is safe to use as a string literal in prompts.
    Returns (is_valid, flags).
    Allows only [a-zA-Z0-9_\\-.]{1,128}.
    """
    flags: list[dict] = []

    if not report_id or not report_id.strip():
        flags.append({
            "check": "REPORT_ID_EMPTY",
            "field": "report_id",
            "severity": "warn",
            "detail": "report_id is empty or whitespace",
            "truncated_sample": "",
        })
        return False, flags

    if len(report_id) > 128:
        flags.append({
            "check": "REPORT_ID_TOO_LONG",
            "field": "report_id",
            "severity": "warn",
            "detail": f"report_id length {len(report_id)} exceeds 128 chars",
            "truncated_sample": report_id[:80],
        })
        return False, flags

    if not _REPORT_ID_SAFE.match(report_id):
        flags.append({
            "check": "REPORT_ID_INVALID_CHARS",
            "field": "report_id",
            "severity": "warn",
            "detail": "report_id contains characters outside [a-zA-Z0-9_\\-.]",
            "truncated_sample": report_id[:80],
        })
        return False, flags

    return True, flags
