"""
risk_assessment_node — final safety net before the report result is returned.

Runs AFTER final_score_node and may override the `action` it produced.

Purpose: prevent dismissing a real threat purely on hoax_probability.
The four override rules below encode the principle that:
  "false negatives (missed threats) are worse than false positives (false alarms)"

Override rules (evaluated in order; first match wins the action override):

  Rule 1 — High urgency + uncertain hoax signal
    if threat_level > 0.7 AND hoax_probability > 0.4:
        action = "human_review"
        false_negative_risk = "high"
        reason = "High urgency + uncertain hoax signal — human required"

  Rule 2 — All-anonymous cluster with critical incident type
    if cluster.anon_ratio == 1.0 AND urgency_level in {"HIGH", "CRITICAL"}:
        action = "escalate"
        false_negative_risk = "high"
        reason = "Fully anonymous cluster with critical incident type"

  Rule 3 — Burst submission + elevated threat
    if rate_limit_flagged AND threat_level > 0.6:
        action = "human_review"
        false_negative_risk = "medium"
        reason = "Burst submission pattern with elevated threat"

  Rule 4 — EXIF conflicts (amplify only if other hoax signals also present)
    if image_metadata_conflicts:
        if hoax_score > 0.5 OR rate_limit_flagged:
            hoax_probability += 0.12  (clipped to 1.0)
            escalation_reason += "; EXIF conflicts corroborate hoax signal"
        else:
            escalation_reason += "; EXIF conflicts detected (possible proxy/old image)"

Appends a final audit entry and returns updated state fields.
"""
from __future__ import annotations

from datetime import datetime, timezone


_CRITICAL_URGENCY_LEVELS = {"HIGH", "CRITICAL"}


def _apply_rule1(
    threat_level: float,
    hoax_probability: float,
    false_negative_risk: str,
    action: str,
    reasons: list[str],
) -> tuple[str, str]:
    """High urgency + uncertain hoax → human_review."""
    if threat_level > 0.7 and hoax_probability > 0.4:
        action = "human_review"
        false_negative_risk = "high"
        reasons.append(
            "High urgency + uncertain hoax signal — human review required"
        )
    return action, false_negative_risk


def _apply_rule2(
    cluster_summary: dict,
    urgency_level: str,
    false_negative_risk: str,
    action: str,
    reasons: list[str],
) -> tuple[str, str]:
    """All-anonymous cluster + critical incident type → escalate."""
    anon_ratio = cluster_summary.get("anon_ratio", 0.0)
    if anon_ratio == 1.0 and urgency_level in _CRITICAL_URGENCY_LEVELS:
        action = "escalate"
        false_negative_risk = "high"
        reasons.append(
            "Fully anonymous cluster with critical incident type"
        )
    return action, false_negative_risk


def _apply_rule3(
    rate_limit_flagged: bool,
    threat_level: float,
    false_negative_risk: str,
    action: str,
    reasons: list[str],
) -> tuple[str, str]:
    """Burst submission + elevated threat → human_review."""
    if rate_limit_flagged and threat_level > 0.6:
        action = "human_review"
        false_negative_risk = false_negative_risk or "medium"
        reasons.append(
            "Burst submission pattern with elevated threat"
        )
    return action, false_negative_risk


def _apply_rule4_exif(
    image_metadata_conflicts: list,
    hoax_score: float,
    rate_limit_flagged: bool,
    hoax_probability: float,
    reasons: list[str],
) -> float:
    """
    EXIF conflicts: amplify hoax_probability only if other signals are present.
    Lone EXIF mismatch could be a proxy reporter or old photo — don't auto-penalise.
    """
    if not image_metadata_conflicts:
        return hoax_probability

    conflict_types = [c.get("type", "unknown") for c in image_metadata_conflicts]
    conflict_label = ", ".join(conflict_types)

    if hoax_score > 0.5 or rate_limit_flagged:
        hoax_probability = min(1.0, hoax_probability + 0.12)
        reasons.append(
            f"EXIF conflicts ({conflict_label}) corroborate hoax signal — "
            "hoax_probability increased by 0.12"
        )
    else:
        reasons.append(
            f"EXIF conflicts ({conflict_label}) detected "
            "(possible proxy reporter or old image — no penalty applied)"
        )
    return hoax_probability


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    hoax_probability: float = float(state.get("hoax_probability", 0.0))
    hoax_score: float = float(state.get("hoax_score", 0.0))
    threat_level: float = float(state.get("threat_level", 0.0))
    urgency_level: str = state.get("urgency_level", "MINIMAL")
    action: str = state.get("action", "monitor")
    cluster_summary: dict = state.get("cluster_summary", {})
    rate_limit_flagged: bool = bool(state.get("rate_limit_flagged", False))
    image_metadata_conflicts: list = state.get("image_metadata_conflicts", [])

    false_negative_risk = "low"
    reasons: list[str] = []

    # Rule 1 — evaluated first; may be superseded by Rule 2
    action, false_negative_risk = _apply_rule1(
        threat_level, hoax_probability, false_negative_risk, action, reasons
    )

    # Rule 2 — can override Rule 1's action (escalate > human_review)
    action, false_negative_risk = _apply_rule2(
        cluster_summary, urgency_level, false_negative_risk, action, reasons
    )

    # Rule 3 — only fires if Rules 1+2 haven't already set human_review or escalate
    if action not in ("human_review", "escalate"):
        action, false_negative_risk = _apply_rule3(
            rate_limit_flagged, threat_level, false_negative_risk, action, reasons
        )

    # Rule 4 — EXIF conflicts (always evaluated; may bump hoax_probability)
    hoax_probability = _apply_rule4_exif(
        image_metadata_conflicts,
        hoax_score,
        rate_limit_flagged,
        hoax_probability,
        reasons,
    )

    escalation_reason = "; ".join(reasons) if reasons else ""

    audit_entry = {
        "node": "risk_assessment_node",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "false_negative_risk": false_negative_risk,
        "escalation_reason": escalation_reason,
        "hoax_probability_final": hoax_probability,
        "rules_triggered": len(reasons),
    }

    return {
        "action": action,
        "hoax_probability": hoax_probability,
        "false_negative_risk": false_negative_risk,
        "escalation_reason": escalation_reason,
        "audit_trail": [audit_entry],
    }
