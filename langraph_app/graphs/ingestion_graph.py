"""
IngestionGraph — embed + classify + index a validated report into Elasticsearch.

Entry point: transform_node
  transform_node → [embed_node, classify_node]  (parallel fan-out)
  embed_node     → index_node                   (fan-in — waits for both)
  classify_node  → index_node
  index_node     → END

No guardrails node — receives already-validated data from IntakeGraph output.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from langraph_app.state import IngestionState
from langraph_app.nodes import (
    transform,
    embed,
    classify,
    index,
)


def build() -> StateGraph:
    graph = StateGraph(IngestionState)

    graph.add_node("transform_node", transform.run)
    graph.add_node("embed_node", embed.run)
    graph.add_node("classify_node", classify.run)
    graph.add_node("index_node", index.run)

    graph.set_entry_point("transform_node")

    # Fan-out: transform → embed + classify in parallel
    graph.add_edge("transform_node", "embed_node")
    graph.add_edge("transform_node", "classify_node")

    # Fan-in: both must complete before index
    graph.add_edge("embed_node", "index_node")
    graph.add_edge("classify_node", "index_node")

    graph.add_edge("index_node", END)

    return graph


ingestion_graph = build().compile()
