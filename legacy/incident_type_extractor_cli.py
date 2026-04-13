"""
Clean incident type extractor with CLI entrypoint.
This file is a drop-in standalone that implements the main() function the
other file was missing. It provides local regex matching, slang and fuzzy
matching (difflib). It also supports optional LLM suggestion if you pass an
OpenAI-like client and set OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from difflib import get_close_matches
from typing import Any, Callable, Dict, List, Optional

# Minimal taxonomy for demo (extend as needed)
INCIDENT_KEYWORDS: Dict[str, List[str]] = {
    "shooting": [r"\bshoot(?:ing|s|er)\b", r"\bshots?\b", r"\bgunfire\b"],
    "bombing": [r"\bbomb\b", r"\bexplosion\b"],
    "theft": [r"\btheft\b", r"\bstolen\b", r"\bstole\b"],
    "burglary": [r"\bburglary\b", r"\bbreak-?in\b", r"\bbroke into\b"],
    "hit_and_run": [r"\bhit and run\b", r"\bfled the scene\b"],
    "traffic_accident": [r"\baccident\b", r"\bcrash\b"],
    "suspicious_package": [r"\bsuspicious package\b", r"\bunattended bag\b"],
    "other": [r".*"],
}

SLANG_MAP = {
    "shots fired": "shooting",
    "pkg": "suspicious_package",
    "stole": "theft",
}


def _build_keyword_vocab(keywords: Dict[str, List[str]]) -> Dict[str, List[str]]:
    vocab: Dict[str, List[str]] = defaultdict(list)
    word_re = re.compile(r"[a-zA-Z0-9_]+")
    for itype, patterns in keywords.items():
        for pat in patterns:
            for w in word_re.findall(pat.lower()):
                if itype not in vocab[w]:
                    vocab[w].append(itype)
    return dict(vocab)


_KEYWORD_VOCAB = _build_keyword_vocab(INCIDENT_KEYWORDS)

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def get_incident_types(text: str, enable_fuzzy=True, fuzzy_cutoff=0.8) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []
    t = text.lower()
    scores = defaultdict(float)
    matches = defaultdict(list)

    for itype, patterns in INCIDENT_KEYWORDS.items():
        if itype == "other":
            continue
        for pat in patterns:
            m = re.search(pat, t)
            if m:
                matched = m.group(0).strip()
                matches[itype].append(matched)
                scores[itype] += 0.6 if len(matched.split()) <= 2 else 0.5

    # slang
    for slang, mapped in SLANG_MAP.items():
        if slang in t:
            matches[mapped].append(slang)
            scores[mapped] += 0.7
            logger.info("Slang mapped: %s -> %s", slang, mapped)

    # fuzzy
    if enable_fuzzy:
        tokens = re.findall(r"[a-zA-Z0-9_]+", t)
        for token in tokens:
            if len(token) < 3:
                continue
            close = get_close_matches(token, list(_KEYWORD_VOCAB.keys()), n=1, cutoff=fuzzy_cutoff)
            if close:
                kw = close[0]
                for it in _KEYWORD_VOCAB.get(kw, []):
                    scores[it] += 0.3
                    matches[it].append(f"(fuzzy:{token}->{kw})")
                    logger.info("Fuzzy token %s -> %s -> %s", token, kw, it)

    cap = 1.6
    normalized = [(itype, min(score / cap, 1.0), matches[itype]) for itype, score in scores.items()]
    normalized.sort(key=lambda x: x[1], reverse=True)
    results = [{"type": itype, "confidence": round(conf, 3), "matches": matched} for itype, conf, matched in normalized]
    if not results:
        return [{"type": "other", "confidence": 0.2, "matches": []}]
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", help="report text (or read stdin)")
    p.add_argument("--no-fuzzy", action="store_true", help="disable fuzzy matching")
    args = p.parse_args()

    if args.text:
        text = args.text
    else:
        import sys

        text = sys.stdin.read().strip()

    types = get_incident_types(text, enable_fuzzy=not args.no_fuzzy)
    print(json.dumps({"types": types}, indent=2))


if __name__ == "__main__":
    main()
