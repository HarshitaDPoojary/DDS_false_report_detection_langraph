"""
final_score_node — combines hoax_score and urgency_score into the system's
primary outputs: hoax_probability, threat_level, action, confidence_range,
and ai_analysis (human-readable explanation for the analyst).

Formulas:
  hoax_probability = clip(hoax_score × (1 - 0.3 × urgency_score), 0, 1)
  threat_level     = clip(urgency_score, 0, 1)   # independent of hoax

Action routing:
  threat_level >= 0.8 AND hoax_probability >= 0.5  → "human_review"
  threat_level >= 0.8 AND hoax_probability < 0.5   → "escalate"
  hoax_probability >= 0.7                          → "dismiss"
  else                                             → "monitor"

Confidence margin:
  margin = 0.15 if cluster_size < 3 else 0.08
  confidence_range = [max(0, hp - margin), min(1, hp + margin)]

ai_analysis is a template-based prose summary of the key factors that
drove the hoax probability — shown to the analyst reviewing the report.
"""
from __future__ import annotations

import datetime as _dt
from datetime import timezone

import numpy as np


# ---------------------------------------------------------------------------
# Action routing thresholds
# ---------------------------------------------------------------------------
_THREAT_HIGH = 0.8
_HOAX_DISMISS = 0.7
_HOAX_HUMAN_REVIEW = 0.5


def _determine_action(hoax_probability: float, threat_level: float) -> str:
    if threat_level >= _THREAT_HIGH and hoax_probability >= _HOAX_HUMAN_REVIEW:
        return "human_review"
    if threat_level >= _THREAT_HIGH:
        return "escalate"
    if hoax_probability >= _HOAX_DISMISS:
        return "dismiss"
    return "monitor"


def _confidence_margin(cluster_size: int) -> float:
    return 0.15 if cluster_size < 3 else 0.08


def _hoax_level(hoax_probability: float) -> str:
    if hoax_probability >= 0.8:
        return "VERY HIGH"
    if hoax_probability >= 0.6:
        return "HIGH"
    if hoax_probability >= 0.4:
        return "MODERATE"
    if hoax_probability >= 0.2:
        return "LOW"
    return "VERY LOW"


def _cluster_lines(cluster_summary: dict) -> list[str]:
    lines: list[str] = []
    cluster_size = cluster_summary.get("cluster_size", 0)
    avg_sim = cluster_summary.get("avg_similarity", 0.0)
    anon_ratio = cluster_summary.get("anon_ratio", 0.0)
    named_ratio = cluster_summary.get("named_ratio", 0.0)
    near_dupes = cluster_summary.get("near_duplicate_count", 0)

    if cluster_size == 0:
        lines.append("• No similar recent reports found in the area — no cluster signal available")
        return lines

    sim_pct = int(avg_sim * 100)
    lines.append(
        f"• {cluster_size} similar report(s) found in the search area "
        f"(avg similarity {sim_pct}%)"
    )
    if anon_ratio > 0:
        anon_count = int(round(anon_ratio * cluster_size))
        lines.append(
            f"  — {anon_count} of {cluster_size} cluster reports are anonymous "
            f"(anonymous ratio {int(anon_ratio*100)}% raises hoax signal)"
        )
    if named_ratio > 0:
        named_count = int(round(named_ratio * cluster_size))
        lines.append(
            f"  — {named_count} of {cluster_size} cluster reports have named witnesses "
            f"(reduces hoax signal)"
        )
    if near_dupes > 0:
        lines.append(
            f"  — {near_dupes} near-duplicate report(s) detected "
            f"(high similarity suggests template reuse)"
        )
    return lines


_BURST_REASON_MAP = {
    "device_burst":     "multiple submissions from same device",
    "geo_burst":        "≥3 similar reports from same area within 1 hour",
    "text_clone_burst": "near-identical text submitted multiple times",
}

_ACTION_RATIONALE = {
    "dismiss":      "Hoax probability is high and urgency is low — safe to dismiss pending analyst review.",
    "monitor":      "Moderate signal — keep on radar but no immediate escalation required.",
    "human_review": "High urgency combined with uncertain hoax signal — human review required before any action.",
    "escalate":     "Low hoax probability and high urgency — treat as credible threat.",
}


def _build_ai_analysis(
    hoax_probability: float,
    urgency_score: float,
    urgency_level: str,
    cluster_summary: dict,
    urgency_breakdown: list,
    reporter_credibility_score: float,
    is_anonymous_reporter: bool,
    rate_limit_flagged: bool,
    rate_limit_reason: str,
    burst_count: int,
    image_metadata_conflicts: list,
    action: str,
) -> str:
    """
    Build a human-readable prose explanation of what drove the hoax probability.
    This is shown to the analyst reviewing the report.
    """
    lines: list[str] = []

    # ── Headline ──────────────────────────────────────────────────────────────
    hp_pct = int(hoax_probability * 100)
    level = _hoax_level(hoax_probability)
    lines.append(f"Hoax probability: {hp_pct}% ({level})")
    lines.append("")

    # ── Cluster analysis ──────────────────────────────────────────────────────
    lines.extend(_cluster_lines(cluster_summary))

    # ── Urgency ───────────────────────────────────────────────────────────────
    urgency_pct = int(urgency_score * 100)
    softening = int(0.3 * urgency_score * 100)
    lines.append(
        f"• Urgency: {urgency_level} ({urgency_pct}%) — "
        f"softens hoax signal by {softening}% (high urgency = treat cautiously)"
    )

    # Top urgency factor
    if urgency_breakdown:
        top = max(urgency_breakdown, key=lambda x: x.get("value", 0.0))
        lines.append(f"  — Primary urgency factor: {top.get('component', '?')} "
                     f"({top.get('details', '')})")

    # ── Reporter credibility ──────────────────────────────────────────────────
    cred_pct = int(reporter_credibility_score * 100)
    if is_anonymous_reporter:
        lines.append(
            f"• Reporter: Anonymous (credibility {cred_pct}% — identity unverifiable)"
        )
    else:
        lines.append(f"• Reporter credibility: {cred_pct}%")

    # ── Rate-limit / burst ────────────────────────────────────────────────────
    if rate_limit_flagged:
        reason_str = _BURST_REASON_MAP.get(rate_limit_reason, rate_limit_reason)
        lines.append(
            f"• Burst/spam flag: {reason_str} "
            f"(burst count: {burst_count}) — possible coordinated false reporting"
        )

    # ── EXIF conflicts ────────────────────────────────────────────────────────
    if image_metadata_conflicts:
        conflict_types = [c.get("type", "unknown") for c in image_metadata_conflicts]
        lines.append(
            f"• Image metadata: {len(image_metadata_conflicts)} EXIF conflict(s) detected "
            f"({', '.join(conflict_types)}) — see audit trail for details"
        )

    # ── Action rationale ──────────────────────────────────────────────────────
    lines.append("")
    lines.append(
        f"Recommended action: {action.upper()} — "
        f"{_ACTION_RATIONALE.get(action, '')}"
    )

    return "\n".join(lines)


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    hoax_score    = float(state.get("hoax_score",    0.0))
    urgency_score = float(state.get("urgency_score", 0.0))

    # ── Core scores ───────────────────────────────────────────────────────────
    raw_hoax = hoax_score * (1.0 - 0.3 * urgency_score)
    hoax_probability = float(np.clip(raw_hoax, 0.0, 1.0))
    threat_level     = float(np.clip(urgency_score, 0.0, 1.0))

    # ── Action ────────────────────────────────────────────────────────────────
    action = _determine_action(hoax_probability, threat_level)

    # ── Confidence range ──────────────────────────────────────────────────────
    cluster_summary: dict = state.get("cluster_summary", {})
    cluster_size = cluster_summary.get("cluster_size", 0)
    margin = _confidence_margin(cluster_size)
    confidence_range = [
        float(np.clip(hoax_probability - margin, 0.0, 1.0)),
        float(np.clip(hoax_probability + margin, 0.0, 1.0)),
    ]

    # ── ai_analysis ───────────────────────────────────────────────────────────
    ai_analysis = _build_ai_analysis(
        hoax_probability=hoax_probability,
        urgency_score=urgency_score,
        urgency_level=state.get("urgency_level", "MINIMAL"),
        cluster_summary=cluster_summary,
        urgency_breakdown=state.get("urgency_breakdown", []),
        reporter_credibility_score=float(state.get("reporter_credibility_score", 0.5)),
        is_anonymous_reporter=bool(state.get("is_anonymous_reporter", False)),
        rate_limit_flagged=bool(state.get("rate_limit_flagged", False)),
        rate_limit_reason=state.get("rate_limit_reason", ""),
        burst_count=int(state.get("burst_count", 0)),
        image_metadata_conflicts=state.get("image_metadata_conflicts", []),
        action=action,
    )

    # ── Full analysis_result dict (validated through Pydantic at API layer) ───
    analysis_result = {
        "report_id":                  state.get("report_id", ""),
        "hoax_probability":           hoax_probability,
        "threat_level":               threat_level,
        "action":                     action,
        "confidence_range":           confidence_range,
        "ai_analysis":                ai_analysis,
        "hoax_score":                 hoax_score,
        "urgency_score":              urgency_score,
        "urgency_level":              state.get("urgency_level", "MINIMAL"),
        "urgency_breakdown":          state.get("urgency_breakdown", []),
        "cluster_summary":            cluster_summary,
        "incident_types":             state.get("incident_types", []),
        "severity":                   state.get("severity", "low"),
        "scored_results":             state.get("scored_results", [])[:10],
        "reporter_credibility_score": float(state.get("reporter_credibility_score", 0.5)),
        "is_anonymous_reporter":      bool(state.get("is_anonymous_reporter", False)),
        "rate_limit_flagged":         bool(state.get("rate_limit_flagged", False)),
        "rate_limit_reason":          state.get("rate_limit_reason", ""),
        "burst_count":                int(state.get("burst_count", 0)),
        "image_metadata_conflicts":   state.get("image_metadata_conflicts", []),
        "guardrail_flags":            state.get("guardrail_flags", []),
        "radius_miles":               state.get("radius_miles", 5.0),
        "lookback_hours":             state.get("lookback_hours", 24.0),
        "effective_radius_miles":     state.get("effective_radius_miles", 5.0),
        "effective_lookback_hours":   state.get("effective_lookback_hours", 24.0),
    }

    audit_entry = {
        "node": "final_score_node",
        "timestamp": _dt.datetime.now(timezone.utc).isoformat(),
        "hoax_probability": hoax_probability,
        "threat_level": threat_level,
        "action": action,
        "confidence_range": confidence_range,
    }

    return {
        "hoax_probability": hoax_probability,
        "threat_level":     threat_level,
        "action":           action,
        "confidence_range": confidence_range,
        "ai_analysis":      ai_analysis,
        "analysis_result":  analysis_result,
        "audit_trail":      [audit_entry],
    }
