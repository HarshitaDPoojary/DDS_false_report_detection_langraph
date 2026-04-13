"""
retrieve_node — searches Elasticsearch for similar reports.

Uses ElasticsearchStore (langchain-elasticsearch) with:
  - ApproxRetrievalStrategy for kNN on text_embedding field
  - dynamic custom_query closure injecting geo + time + type filters
    built by the legacy query-builder functions

Returns: {candidate_hits, has_candidates}
  candidate_hits: list of ES hit dicts  {_source, _score, page_content}
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore, ApproxRetrievalStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache

from legacy.es_query_builder import build_geo_query, build_time_query, build_incident_type_query
from langraph_app.config.settings import get_settings
from langraph_app.nodes.embed import _get_model as _get_embeddings


@lru_cache(maxsize=1)
def _get_es_client() -> Elasticsearch:
    s = get_settings()
    return Elasticsearch(s.es_host, api_key=s.es_api_key)


def run(state: dict) -> dict:
    settings   = get_settings()
    location   = state.get("location", [])       # [lon, lat]
    time_start = state.get("time_start", "")
    time_end   = state.get("time_end", "")
    incident_types = state.get("incident_types", [])
    radius_miles   = state.get("radius_miles", 5.0)
    lookback_hours = state.get("lookback_hours", 24.0)
    report_id      = state.get("report_id", "")

    if len(location) < 2:
        return {"candidate_hits": [], "has_candidates": False}

    lon, lat = location[0], location[1]

    # Build filter components from legacy query builder
    geo_filter  = build_geo_query(lat, lon, radius_miles)
    time_filter = build_time_query(time_start, time_end, lookback_hours=lookback_hours)
    primary_type = incident_types[0]["type"] if incident_types else "other"
    type_query   = build_incident_type_query(primary_type)

    # Exclusion clause: skip the query report itself
    must_not = [{"term": {"report_id": report_id}}] if report_id else []

    # Build closure that injects filters into the kNN query body
    def custom_query(query_body: dict, query: str) -> dict:
        query_body["query"] = {
            "bool": {
                "filter":   [geo_filter, time_filter],
                "should":   [type_query],
                "must_not": must_not,
            }
        }
        return query_body

    store = ElasticsearchStore(
        index_name=settings.es_index,
        embedding=_get_embeddings(),
        es_connection=_get_es_client(),
        strategy=ApproxRetrievalStrategy(num_candidates=100),
        custom_query=custom_query,
    )

    try:
        docs_and_scores = store.similarity_search_with_score(
            state.get("free_text", ""), k=20
        )
    except Exception as e:
        return {"candidate_hits": [], "has_candidates": False,
                "error": f"retrieve_node ES error: {e}"}

    hits = [
        {
            "_source":      doc.metadata,
            "_score":       float(score),
            "page_content": doc.page_content,
        }
        for doc, score in docs_and_scores
    ]

    return {
        "candidate_hits": hits,
        "has_candidates": len(hits) > 0,
    }
