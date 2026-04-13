"""
text_utils — pure text helper functions with no heavy dependencies.

Isolated here so validate_node never imports legacy/gpt_incident_agent.py,
which instantiates an OpenAI client at module level and requires OPENAI_API_KEY.
"""
from __future__ import annotations

import hashlib
import os
import re

_SOC_SALT = os.environ.get("SOC_SALT", "pepper")


def hash_soc(name: str, org: str, dob_fragment: str = "") -> str:
    """SHA-256 pseudonymous hash of subject-of-concern identity."""
    h = hashlib.sha256()
    h.update(
        (_SOC_SALT + "|" + name.strip().lower() + "|" + org.strip().lower()
         + "|" + dob_fragment.strip()).encode("utf-8")
    )
    return h.hexdigest()[:24]


def quick_quotes(text: str) -> list[str]:
    """Extract verbatim quoted speech and threat-like phrases from free text."""
    quotes: list[str] = []
    for l, r in [("'", "'"), ('"', '"'), ("\u201c", "\u201d"), ("\u2018", "\u2019")]:
        pattern = re.escape(l) + r"(.+?)" + re.escape(r)
        for m in re.finditer(pattern, text, re.S):
            q = m.group(1).strip()
            if 3 <= len(q) <= 400:
                quotes.append(q)
    # threat-like fallback
    for m in re.finditer(
        r"\b(will|gonna|going\s+to)\s+(kill|shoot|detonate|burn|hurt)\b.*",
        text,
        re.I,
    ):
        quotes.append(m.group(0)[:200])
    # dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in quotes:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out
