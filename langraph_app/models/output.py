"""
Output schemas for analysis results returned to API/CLI callers.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClusterSummary(BaseModel):
    avg_similarity: float = 0.0
    named_ratio: float = 0.0
    anon_ratio: float = 0.0
    cluster_size: int = 0
    near_duplicate_count: int = 0


class ScoredResult(BaseModel):
    report_id: str = ""
    final_score: float = 0.0
    geo_score: float = 0.0
    time_score: float = 0.0
    text_score: float = 0.0
    type_score: float = 0.0
    distance_miles: float = 0.0
    time_delta_hours: float = 0.0
    incident_type: str = ""
    free_text: str = ""


class AnalysisResult(BaseModel):
    report_id: str = ""

    # ── Primary outputs ───────────────────────────────────────────────────────
    hoax_probability: float = Field(0.0, ge=0.0, le=1.0,
        description="0 = likely genuine, 1 = likely hoax")
    threat_level: float = Field(0.0, ge=0.0, le=1.0,
        description="Independent severity/danger score; orthogonal to hoax_probability")
    action: str = Field("monitor",
        description="Recommended response: dismiss | monitor | human_review | escalate")
    confidence_range: List[float] = Field(default_factory=lambda: [0.0, 0.0],
        description="[low, high] confidence interval for hoax_probability")
    ai_analysis: str = Field("",
        description="Human-readable explanation of factors driving hoax probability (shown to analyst)")

    # ── Component scores ──────────────────────────────────────────────────────
    hoax_score: float = Field(0.0, ge=0.0, le=1.0)
    urgency_score: float = Field(0.0, ge=0.0, le=1.0)
    urgency_level: str = "MINIMAL"      # MINIMAL / LOW / MEDIUM / HIGH / CRITICAL

    # ── Classification ────────────────────────────────────────────────────────
    incident_types: List[Dict[str, Any]] = []
    severity: str = "low"

    # ── Cluster evidence ──────────────────────────────────────────────────────
    cluster_summary: ClusterSummary = Field(default_factory=ClusterSummary)
    scored_results: List[Dict[str, Any]] = []   # top-10 similar reports

    # ── Urgency breakdown (factors) ───────────────────────────────────────────
    urgency_breakdown: List[Dict[str, Any]] = []

    # ── Reporter signals ──────────────────────────────────────────────────────
    reporter_credibility_score: float = Field(0.5, ge=0.0, le=1.0)
    is_anonymous_reporter: bool = False

    # ── Rate-limit / burst detection ─────────────────────────────────────────
    rate_limit_flagged: bool = False
    rate_limit_reason: str = ""
    burst_count: int = 0

    # ── Image signals ─────────────────────────────────────────────────────────
    image_metadata_conflicts: List[Dict[str, Any]] = []

    # ── Risk assessment ───────────────────────────────────────────────────────
    false_negative_risk: str = "low"   # low | medium | high
    escalation_reason: str = ""

    # ── Guardrail flags ───────────────────────────────────────────────────────
    guardrail_flags: List[Dict[str, Any]] = []

    # ── Audit trail ───────────────────────────────────────────────────────────
    audit_trail: List[Dict[str, Any]] = []

    # ── Meta ──────────────────────────────────────────────────────────────────
    radius_miles: float = 5.0
    lookback_hours: float = 24.0
    effective_radius_miles: float = 5.0
    effective_lookback_hours: float = 24.0
