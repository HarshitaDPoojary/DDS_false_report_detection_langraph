"""
TypedDict state schemas for all three LangGraph graphs.

Each graph has its own state class. Nodes return partial dicts
that LangGraph merges into the running state.
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Analysis Graph state
# Tracks a report from raw input through retrieval, scoring, and final output.
# ---------------------------------------------------------------------------
class AnalysisState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    raw_report: dict           # original JSON report dict
    radius_miles: float        # geo search radius (default 5.0)
    lookback_hours: float      # time window lookback (default 24.0)
    scorer_weights: dict       # {geo, time, text, type} — sum to 1.0

    # ── After guardrails_node (FIRST node) ───────────────────────────────────
    guardrail_flags: list              # list of GuardrailFlag dicts
    guardrail_hard_block: bool         # True = abort graph immediately
    guardrail_sanitized_text: str      # sanitized free_text
    guardrail_sanitized_ocr: list      # sanitized OCR strings

    # ── After rate_limit_check_node ──────────────────────────────────────────
    rate_limit_flagged: bool
    rate_limit_reason: str             # "device_burst" | "geo_burst" | "text_clone_burst" | ""
    burst_count: int

    # ── After transform_node ─────────────────────────────────────────────────
    report_id: str
    free_text: str
    location: list             # [lon, lat]  ← ES geo_point order
    time_start: str            # ISO datetime string
    time_end: str
    time_midpoint: str

    # ── After embed_node (parallel branch A) ─────────────────────────────────
    text_embedding: list       # 384-dimensional float list

    # ── After classify_node (parallel branch B) ──────────────────────────────
    incident_types: list       # [{type, confidence, matches}]
    severity: str              # low | medium | high | critical

    # ── After reporter_credibility_node (parallel branch C) ──────────────────
    reporter_credibility_score: float  # 0..1; 0.4=anonymous, higher=verified
    is_anonymous_reporter: bool
    credibility_breakdown: dict

    # ── After retrieve_node ──────────────────────────────────────────────────
    # Annotated[list, operator.add] lets LangGraph merge parallel writes safely
    candidate_hits: Annotated[list, operator.add]
    has_candidates: bool
    effective_radius_miles: float      # actual radius used (may differ from input)
    effective_lookback_hours: float    # actual time window used

    # ── After score_node ─────────────────────────────────────────────────────
    scored_results: list       # [{final_score, geo, time, text, type, result}]

    # ── After hoax_node ──────────────────────────────────────────────────────
    hoax_score: float          # 0..1
    cluster_summary: dict      # avg_similarity, named_ratio, anon_ratio,
                               # cluster_size, near_duplicate_count

    # ── After urgency_node ───────────────────────────────────────────────────
    urgency_score: float       # 0..1 (normalized)
    urgency_level: str         # MINIMAL | LOW | MEDIUM | HIGH | CRITICAL
    urgency_breakdown: list    # [{component, value, details}]

    # ── After final_score_node — PRIMARY OUTPUT ───────────────────────────────
    hoax_probability: float    # 0..1  (higher = more likely a hoax)
    threat_level: float        # 0..1  independent of hoax; based on urgency + severity
    confidence_range: list     # [low, high] interval
    action: str                # "dismiss" | "monitor" | "human_review" | "escalate"
    ai_analysis: str           # human-readable explanation of hoax factors (shown to analyst)
    analysis_result: dict      # full structured output dict

    # ── After risk_assessment_node ───────────────────────────────────────────
    false_negative_risk: str   # "low" | "medium" | "high"
    escalation_reason: str

    # ── Cross-cutting ────────────────────────────────────────────────────────
    audit_trail: Annotated[list, operator.add]   # [{node, timestamp, key_outputs}]

    # ── Error handling ────────────────────────────────────────────────────────
    error: Optional[str]


# ---------------------------------------------------------------------------
# Ingestion Graph state
# Indexes a single report into Elasticsearch.
# ---------------------------------------------------------------------------
class IngestionState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    raw_report: dict

    # ── After guardrails_node (not present in IngestionGraph — skip) ─────────
    # IngestionGraph receives already-validated data from IntakeGraph output.

    # ── After transform_node ─────────────────────────────────────────────────
    report_id: str
    free_text: str
    location: list
    time_start: str
    time_end: str
    time_midpoint: str

    # ── After embed_node (parallel A) ────────────────────────────────────────
    text_embedding: list

    # ── After classify_node (parallel B) ─────────────────────────────────────
    incident_types: list
    severity: str

    # ── After index_node ─────────────────────────────────────────────────────
    indexed: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# Intake Graph state
# Converts raw form submission (text + optional attachments) into a
# structured ExtractionResult ready for ingestion.
# ---------------------------------------------------------------------------
class IntakeState(TypedDict, total=False):
    # ── Input ────────────────────────────────────────────────────────────────
    text_input: str            # free-text from form field
    attachments: list          # list of raw bytes (one per file)
    attachment_names: list     # matching filenames (used to detect type)
    form_metadata: dict        # device, IP, jurisdiction, etc.

    # ── After guardrails_node (FIRST node) ───────────────────────────────────
    guardrail_flags: list
    guardrail_hard_block: bool
    guardrail_sanitized_text: str
    guardrail_sanitized_ocr: list

    # ── After check_attachments_node ─────────────────────────────────────────
    has_attachments: bool

    # ── After ocr_node (only when has_attachments=True) ──────────────────────
    ocr_texts: list            # per-file OCR text strings
    image_data_urls: list      # base64 data: URLs for GPT-4o vision

    # ── After vision_node (parallel with ocr_node) ───────────────────────────
    visual_description: str    # GPT-4o scene/people/vehicle description

    # ── After image_metadata_node (parallel with ocr_node) ───────────────────
    image_metadata: list       # [{gps_lat, gps_lon, timestamp, device}, ...]
    image_metadata_conflicts: list  # [{type, image_file, exif_value, claimed_value, delta}]

    # ── After classify_attachments_node ──────────────────────────────────────
    attachment_types: list     # parallel to attachments; one label per file
                               # "screenshot"|"vehicle"|"id_document"|"person"|"document"|"unknown"
    has_screenshot: bool
    has_vehicle: bool
    has_id_document: bool
    has_person: bool

    # ── After screenshot_node ─────────────────────────────────────────────────
    chat_transcript: list      # [{sender, message, timestamp, platform}]

    # ── After vehicle_node ────────────────────────────────────────────────────
    vehicle_extractions: list  # [{plate, make_model, color, damage_description}]

    # ── After id_document_node ────────────────────────────────────────────────
    id_document_extractions: list  # [{full_name, date_of_birth, address, id_number, ...}]

    # ── After person_node ─────────────────────────────────────────────────────
    person_descriptions: list  # [{appearance, is_suspect, face_match_result}]

    # ── After extract_node ───────────────────────────────────────────────────
    extraction_result: dict    # ExtractionResult.model_dump()

    # ── After validate_node ──────────────────────────────────────────────────
    validated_report: dict     # final validated + enriched report dict
    soc_hash: str              # SHA-256 subject-of-concern hash
    screens_evidence: bool     # True if images/PDFs were processed

    # ── Cross-cutting ────────────────────────────────────────────────────────
    audit_trail: Annotated[list, operator.add]

    # ── Error handling ────────────────────────────────────────────────────────
    error: Optional[str]
