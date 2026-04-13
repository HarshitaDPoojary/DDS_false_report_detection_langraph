"""
Shared constants extracted from legacy modules.
These are pure data — no logic, no imports.
"""
from __future__ import annotations
from typing import Dict, List

# ---------------------------------------------------------------------------
# Canonical incident type taxonomy + regex keyword patterns
# Source: legacy/incident_severity_score.py
# ---------------------------------------------------------------------------
INCIDENT_KEYWORDS: Dict[str, List[str]] = {
    "shooting":           [r"\bshoot(?:ing|s|er)\b", r"\bshots?\b", r"\bgunfire\b", r"\bopen fire\b", r"\bactive shooter\b"],
    "bombing":            [r"\bbomb\b", r"\bexplosion\b", r"\bexplosive\b", r"\bied\b", r"\bsuspicious device\b"],
    "stabbing":           [r"\bstab(?:bing|bed|ber)\b", r"\bknife\b", r"\bcut\b"],
    "assault":            [r"\bassault\b", r"\bbeat(?:ing)?\b", r"\battack(?:ed|s)?\b"],
    "robbery":            [r"\brobber(?:y)?\b", r"\bmugging\b", r"\barmed robbery\b"],
    "burglary":           [r"\bburglary\b", r"\bbreak-?in\b", r"\bhome invasion\b", r"\bbroke into\b", r"\bbroke in\b", r"\bforced (?:entry|open)\b"],
    "theft":              [r"\btheft\b", r"\bstolen\b", r"\bshoplift(?:ing|ed)?\b", r"\bsteal(?:ing|s)?\b", r"\bstole\b", r"\bsnatch(?:ed)?\b", r"\btook\b", r"\brobbed\b"],
    "kidnapping":         [r"\bkidnap(?:ping|ped)?\b", r"\babduct(?:ed|ion)?\b", r"\btaken\b"],
    "vandalism":          [r"\bvandal(?:ism|ize|ised)\b", r"\bgraffiti\b", r"\bproperty damage\b"],
    "arson":              [r"\barson\b", r"\bset (on )?fire\b", r"\bsuspicious fire\b"],
    "hit_and_run":        [r"\bhit and run\b", r"\bhit-and-run\b", r"\bfled the scene\b", r"\bfled\b"],
    "traffic_accident":   [r"\baccident\b", r"\bcrash\b", r"\bcollision\b", r"\bpileup\b"],
    "sexual_assault":     [r"\bsexual assault\b", r"\brape\b"],
    "suspicious_package": [r"\bsuspicious package\b", r"\bunattended bag\b", r"\bbag with wires\b"],
    "protest":            [r"\bprotest\b", r"\briot\b", r"\bdemonstration\b"],
    "hazmat":             [r"\bchemical spill\b", r"\bgas leak\b", r"\bhazardous material\b", r"\bhazmat\b"],
    "hostage":            [r"\bhostage\b", r"\bheld hostage\b"],
    "medical_emergency":  [r"\bcollapsed\b", r"\bnot breathing\b", r"\bunconscious\b", r"\bcardiac arrest\b", r"\bmedical emergency\b"],
    "natural_disaster":   [r"\bearthquake\b", r"\bflood\b", r"\btornado\b", r"\bstorm\b"],
    "cyber_incident":     [r"\bhack(?:ed|ing)?\b", r"\bdata breach\b", r"\bphish(?:ing)?\b"],
}

CANONICAL_TYPES: List[str] = list(INCIDENT_KEYWORDS.keys()) + ["other"]

# ---------------------------------------------------------------------------
# Base urgency score per incident type (0–10 scale)
# Source: legacy/incident_severity_score.py
# ---------------------------------------------------------------------------
BASE_SCORE_BY_TYPE: Dict[str, float] = {
    "shooting":           9.0,
    "bombing":            10.0,
    "stabbing":           6.5,
    "assault":            6.0,
    "robbery":            5.5,
    "burglary":           4.0,
    "theft":              2.0,
    "kidnapping":         9.5,
    "vandalism":          2.0,
    "arson":              8.0,
    "hit_and_run":        6.0,
    "traffic_accident":   5.0,
    "sexual_assault":     9.0,
    "suspicious_package": 8.5,
    "protest":            4.0,
    "hazmat":             9.5,
    "hostage":            10.0,
    "medical_emergency":  7.0,
    "natural_disaster":   10.0,
    "cyber_incident":     3.0,
    "other":              1.0,
}

# ---------------------------------------------------------------------------
# Severity level ordering and type→severity mapping
# Source: legacy/incident_severity_score.py
# ---------------------------------------------------------------------------
SEVERITY_ORDER: List[str] = ["unknown", "low", "medium", "high", "critical"]

TYPE_TO_SEVERITY: Dict[str, str] = {
    "shooting":           "critical",
    "bombing":            "critical",
    "hostage":            "critical",
    "kidnapping":         "critical",
    "hazmat":             "critical",
    "natural_disaster":   "critical",
    "sexual_assault":     "high",
    "arson":              "high",
    "suspicious_package": "high",
    "medical_emergency":  "high",
    "stabbing":           "high",
    "assault":            "high",
    "robbery":            "medium",
    "hit_and_run":        "medium",
    "traffic_accident":   "medium",
    "burglary":           "medium",
    "protest":            "medium",
    "theft":              "low",
    "vandalism":          "low",
    "cyber_incident":     "low",
    "other":              "low",
}

# ---------------------------------------------------------------------------
# Max urgency normalization denominator
# Source: legacy/severity_urgency_score.py
# ---------------------------------------------------------------------------
MAX_URGENCY_POINTS: float = 23.5

# ---------------------------------------------------------------------------
# Incident type classifier prompt (used by classify node LLM fallback)
# ---------------------------------------------------------------------------
CLASSIFICATION_SYSTEM_PROMPT = (
    "You are an incident report classifier for a law enforcement system. "
    "Classify the report into one or more canonical incident types. "
    "Return ONLY valid JSON — no prose, no markdown."
)

CLASSIFICATION_HUMAN_PROMPT = (
    "Canonical types: {canonical_types}\n\n"
    "Rules:\n"
    "- Return a JSON object with a 'types' array.\n"
    "- Each element: {{\"type\": <canonical>, \"confidence\": <0.0-1.0>, \"reason\": <short string>}}\n"
    "- If none apply: [{{\"type\": \"other\", \"confidence\": 0.1, \"reason\": \"no clear indicators\"}}]\n\n"
    "Report:\n{report_text}"
)
