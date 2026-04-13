"""
index_node — indexes a validated report document into Elasticsearch.

Wraps the validated_report dict as a LangChain Document and calls
ElasticsearchStore.add_documents(). The embedding must already be
present in state (added by embed_node).
"""
from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore

from langraph_app.config.settings import get_settings


def _get_store() -> ElasticsearchStore:
    s = get_settings()
    return ElasticsearchStore(
        es_url=s.es_host,
        index_name=s.es_index,
        es_api_key=s.es_api_key or None,
        embedding=None,   # embeddings already pre-computed; we supply them directly
    )


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    validated: dict = state.get("validated_report") or state.get("raw_report", {})
    text_embedding: list = state.get("text_embedding", [])
    free_text: str = state.get("free_text", "")
    report_id: str = state.get("report_id", validated.get("report_id", ""))

    metadata = {
        **validated,
        "report_id":         report_id,
        "location":          state.get("location", []),
        "time_start":        state.get("time_start", ""),
        "time_end":          state.get("time_end", ""),
        "time_midpoint":     state.get("time_midpoint", ""),
        "incident_types":    state.get("incident_types", []),
        "severity":          state.get("severity", "low"),
        "image_metadata":    state.get("image_metadata", []),
        "visual_description": state.get("visual_description", ""),
        "indexed_at":        datetime.now(timezone.utc).isoformat(),
    }

    doc = Document(page_content=free_text, metadata=metadata)

    try:
        store = _get_store()
        store.add_documents(
            documents=[doc],
            ids=[report_id],
            embeddings=[text_embedding] if text_embedding else None,
        )
    except Exception as exc:
        return {"error": f"index_node: Elasticsearch error: {exc}", "indexed": False}

    audit_entry = {
        "node": "index_node",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "report_id": report_id,
        "indexed": True,
    }

    return {"indexed": True, "audit_trail": [audit_entry]}
