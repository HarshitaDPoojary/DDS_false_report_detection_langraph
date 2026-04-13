"""
transform_node — extract structured location + time window from raw_report.

Imports legacy functions to normalize the 3 location formats and 2 time
formats used across the existing data sets.

Returns:
    report_id, free_text, location ([lon, lat]), time_start, time_end,
    time_midpoint — all written into state for downstream nodes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import uuid

from legacy.es_ingest_data import extract_location, extract_time_window


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    raw: dict = state.get("raw_report", {})

    # ── report_id ─────────────────────────────────────────────────────────────
    report_id: str = (
        raw.get("report_id")
        or state.get("report_id")
        or str(uuid.uuid4())
    )

    # ── free_text ─────────────────────────────────────────────────────────────
    # Use guardrail-sanitized text if available; fall back to raw field.
    free_text: str = (
        state.get("guardrail_sanitized_text")
        or raw.get("free_text", "")
    )

    # ── location ──────────────────────────────────────────────────────────────
    try:
        location = extract_location(raw)
    except (ValueError, KeyError, TypeError) as exc:
        return {"error": f"transform_node: location extraction failed: {exc}"}

    # ── time window ───────────────────────────────────────────────────────────
    try:
        time_info = extract_time_window(raw)
    except (ValueError, KeyError, TypeError) as exc:
        return {"error": f"transform_node: time extraction failed: {exc}"}

    audit_entry = {
        "node": "transform_node",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_id": report_id,
        "location": location,
        "time_start": time_info.get("start"),
        "time_end": time_info.get("end"),
    }

    return {
        "report_id": report_id,
        "free_text": free_text,
        "location": location,
        "time_start": time_info.get("start", ""),
        "time_end": time_info.get("end", ""),
        "time_midpoint": time_info.get("midpoint", ""),
        "audit_trail": [audit_entry],
    }
