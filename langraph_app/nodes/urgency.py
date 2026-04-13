"""
urgency_node — context-based urgency scoring.

Wraps legacy calculate_urgency_score() which factors in:
  - incident type base score
  - weapon type, suspect count
  - vulnerability (children, elderly, crowd)
  - time of day, location type (school, hospital, bank…)

Returns: {urgency_score, urgency_level, urgency_breakdown}
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from legacy.severity_urgency_score import calculate_urgency_score, get_urgency_level


def run(state: dict) -> dict:
    free_text      = state.get("free_text", "")
    incident_types = state.get("incident_types", [])

    result = calculate_urgency_score(free_text, incident_types=incident_types)

    score     = float(result.get("score", 0.0))
    level     = get_urgency_level(score)
    breakdown = result.get("breakdown", [])

    return {
        "urgency_score":     score,
        "urgency_level":     level,
        "urgency_breakdown": breakdown,
    }
