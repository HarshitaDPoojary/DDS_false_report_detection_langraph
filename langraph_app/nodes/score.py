"""
score_node — multi-dimensional similarity re-ranking.

Wraps legacy SimilarityScorer which computes:
  geo × 0.2 + time × 0.3 + text × 0.3 + type × 0.2

Returns: {scored_results}  — list sorted by final_score desc
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from legacy.similarity_scoring import SimilarityScorer

_DEFAULT_WEIGHTS = {"geo_weight": 0.2, "time_weight": 0.3,
                    "text_weight": 0.3, "type_weight": 0.2}


def run(state: dict) -> dict:
    raw_report     = state.get("raw_report", {})
    candidate_hits = state.get("candidate_hits", [])
    weights        = state.get("scorer_weights") or {}

    # Build scorer with custom weights or defaults
    scorer = SimilarityScorer(
        geo_weight  = weights.get("geo",  _DEFAULT_WEIGHTS["geo_weight"]),
        time_weight = weights.get("time", _DEFAULT_WEIGHTS["time_weight"]),
        text_weight = weights.get("text", _DEFAULT_WEIGHTS["text_weight"]),
        type_weight = weights.get("type", _DEFAULT_WEIGHTS["type_weight"]),
    )

    # Enrich the query report with transformed fields so scorer can access them
    query_report = {
        **raw_report,
        "location":   state.get("location", []),
        "time_start": state.get("time_start", ""),
        "time_end":   state.get("time_end", ""),
    }

    scored = scorer.rank_results(query_report, candidate_hits)
    return {"scored_results": scored}
