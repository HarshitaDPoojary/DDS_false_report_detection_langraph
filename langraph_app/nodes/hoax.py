"""
hoax_node — cluster-based hoax detection.

Wraps legacy score_similarity_and_hoax() which computes:
  - cosine similarity between new report and candidate texts
  - anonymity ratio of cluster (anon reporters → higher hoax signal)
  - near-duplicate detection (similarity > 0.90)

Returns: {hoax_score, cluster_summary}
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from legacy.get_similarity_hoax import score_similarity_and_hoax


def run(state: dict) -> dict:
    free_text      = state.get("free_text", "")
    scored_results = state.get("scored_results", [])
    report_id      = state.get("report_id", "")

    # Build new_report dict in the shape expected by legacy function
    new_report = {
        "report_id": report_id,
        "text":      free_text,
    }

    # Convert scored_results → candidate_reports list expected by legacy fn
    # Each item needs: report_id, text, is_anonymous
    candidate_reports = []
    for item in scored_results:
        src = item.get("result", {})
        reporter = src.get("reporter", {}) or {}
        is_anon  = reporter.get("anonymous", True)
        text     = src.get("free_text", item.get("page_content", ""))
        candidate_reports.append({
            "report_id":   src.get("report_id", ""),
            "text":        text,
            "is_anonymous": bool(is_anon),
        })

    if not candidate_reports:
        return {
            "hoax_score": 0.0,
            "cluster_summary": {
                "avg_similarity": 0.0, "named_ratio": 0.0,
                "anon_ratio": 0.0, "cluster_size": 0,
                "near_duplicate_count": 0,
            },
        }

    result = score_similarity_and_hoax(new_report, candidate_reports)

    return {
        "hoax_score":      float(result.get("hoax_score", 0.0)),
        "cluster_summary": result.get("summary", {}),
    }
