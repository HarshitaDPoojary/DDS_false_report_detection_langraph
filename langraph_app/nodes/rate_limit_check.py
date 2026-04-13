"""
rate_limit_check_node — burst/spam detection for coordinated false reporting.

Works for fully anonymous submissions — uses signal fingerprints, not identity:
  1. device_hash (from raw_report.reporter.device_hash) — strongest, not tied to name
  2. geo_burst: ≥3 same incident_type within 5-mile radius in last hour
  3. text_clone_burst: cosine similarity >0.85 against recent reports in same geo

Does NOT block — flags are inputs to hoax_node and risk_assessment_node.

Uses the direct Elasticsearch client (not LangChain store) for term/range queries.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from elasticsearch import Elasticsearch

from langraph_app.config.settings import get_settings

_BURST_RADIUS_MILES = 5.0
_BURST_WINDOW_MINUTES = 60
_BURST_THRESHOLD = 3
_TEXT_CLONE_SIM = 0.85


def _get_es() -> Elasticsearch:
    s = get_settings()
    kwargs: dict = {"hosts": [s.es_host]}
    if s.es_api_key:
        kwargs["api_key"] = s.es_api_key
    return Elasticsearch(**kwargs)


def _since_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=_BURST_WINDOW_MINUTES)).isoformat()


def _check_device_burst(es: Elasticsearch, index: str, device_hash: str) -> int:
    """Count recent submissions from the same device hash."""
    if not device_hash:
        return 0
    try:
        resp = es.count(index=index, body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"reporter.device_hash": device_hash}},
                        {"range": {"indexed_at": {"gte": _since_iso()}}},
                    ]
                }
            }
        })
        return int(resp.get("count", 0))
    except Exception:
        return 0


def _check_geo_burst(
    es: Elasticsearch,
    index: str,
    location: list,
    incident_type: str,
) -> int:
    """Count recent same-type reports within burst radius."""
    if not location or len(location) < 2:
        return 0
    try:
        lon, lat = location[0], location[1]
        resp = es.count(index=index, body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"incident_types.type": incident_type}},
                        {"range": {"indexed_at": {"gte": _since_iso()}}},
                        {
                            "geo_distance": {
                                "distance": f"{_BURST_RADIUS_MILES}mi",
                                "location": {"lat": lat, "lon": lon},
                            }
                        },
                    ]
                }
            }
        })
        return int(resp.get("count", 0))
    except Exception:
        return 0


def _check_text_clone(
    es: Elasticsearch,
    index: str,
    text_embedding: list,
    location: list,
) -> bool:
    """Check for near-identical text (cosine sim > 0.85) in same geo within last hour."""
    if not text_embedding or not location or len(location) < 2:
        return False
    try:
        lon, lat = location[0], location[1]
        resp = es.search(index=index, body={
            "query": {
                "bool": {
                    "must": [
                        {"range": {"indexed_at": {"gte": _since_iso()}}},
                        {
                            "geo_distance": {
                                "distance": f"{_BURST_RADIUS_MILES}mi",
                                "location": {"lat": lat, "lon": lon},
                            }
                        },
                    ]
                }
            },
            "knn": {
                "field": "text_embedding",
                "query_vector": text_embedding,
                "k": 1,
                "num_candidates": 10,
            },
            "_source": False,
            "size": 1,
        })
        hits = resp.get("hits", {}).get("hits", [])
        if hits:
            score = hits[0].get("_score", 0.0)
            # ES cosine similarity scores: 1.0 = identical; threshold mapped from 0.85
            return score >= _TEXT_CLONE_SIM
    except Exception:
        pass
    return False


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    s = get_settings()
    reporter: dict = (state.get("raw_report") or {}).get("reporter", {})
    device_hash: str = reporter.get("device_hash", "")
    location: list = state.get("location", [])
    text_embedding: list = state.get("text_embedding", [])
    incident_types: list = state.get("incident_types", [])
    primary_type = incident_types[0].get("type", "other") if incident_types else "other"

    try:
        es = _get_es()
        index = s.es_index

        # Check 1: device burst
        device_count = _check_device_burst(es, index, device_hash)
        if device_count >= _BURST_THRESHOLD:
            return {
                "rate_limit_flagged": True,
                "rate_limit_reason": "device_burst",
                "burst_count": device_count,
            }

        # Check 2: geo burst
        geo_count = _check_geo_burst(es, index, location, primary_type)
        if geo_count >= _BURST_THRESHOLD:
            return {
                "rate_limit_flagged": True,
                "rate_limit_reason": "geo_burst",
                "burst_count": geo_count,
            }

        # Check 3: text clone burst
        if _check_text_clone(es, index, text_embedding, location):
            return {
                "rate_limit_flagged": True,
                "rate_limit_reason": "text_clone_burst",
                "burst_count": 1,
            }

    except Exception:
        # Rate limit check failure is non-fatal — continue without flagging
        pass

    return {
        "rate_limit_flagged": False,
        "rate_limit_reason": "",
        "burst_count": 0,
    }
