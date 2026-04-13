"""
AnalysisGraph — hoax detection pipeline.

Flow:
  guardrails_node          ← FIRST node; aborts on hard block
  rate_limit_check_node
  transform_node
    ├── embed_node          (parallel)
    ├── classify_node       (parallel)
    └── reporter_credibility_node (parallel)
  retrieve_node             (fan-in; waits for all 3)
    ├── score_node          (if has_candidates)
    └── urgency_node        (if no candidates — skip score+hoax)
  score_node → hoax_node → urgency_node
  urgency_node → final_score_node
  final_score_node → risk_assessment_node → END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from langraph_app.state import AnalysisState
from langraph_app.nodes import (
    guardrails,
    rate_limit_check,
    transform,
    embed,
    classify,
    reporter_credibility,
    retrieve,
    score,
    hoax,
    urgency,
    final_score,
    risk_assessment,
)


def _route_after_guardrails(state: AnalysisState) -> str:
    return "abort" if state.get("guardrail_hard_block") else "continue"


def _route_after_retrieve(state: AnalysisState) -> str:
    return "score_node" if state.get("has_candidates") else "urgency_node"


def build() -> StateGraph:
    graph = StateGraph(AnalysisState)

    graph.add_node("guardrails_node", guardrails.run)
    graph.add_node("rate_limit_check_node", rate_limit_check.run)
    graph.add_node("transform_node", transform.run)
    graph.add_node("embed_node", embed.run)
    graph.add_node("classify_node", classify.run)
    graph.add_node("reporter_credibility_node", reporter_credibility.run)
    graph.add_node("retrieve_node", retrieve.run)
    graph.add_node("score_node", score.run)
    graph.add_node("hoax_node", hoax.run)
    graph.add_node("urgency_node", urgency.run)
    graph.add_node("final_score_node", final_score.run)
    graph.add_node("risk_assessment_node", risk_assessment.run)

    graph.set_entry_point("guardrails_node")

    graph.add_conditional_edges(
        "guardrails_node",
        _route_after_guardrails,
        {"abort": END, "continue": "rate_limit_check_node"},
    )

    graph.add_edge("rate_limit_check_node", "transform_node")

    # Fan-out: transform → embed + classify + reporter_credibility in parallel
    graph.add_edge("transform_node", "embed_node")
    graph.add_edge("transform_node", "classify_node")
    graph.add_edge("transform_node", "reporter_credibility_node")

    # Fan-in: all 3 must complete before retrieve
    graph.add_edge("embed_node", "retrieve_node")
    graph.add_edge("classify_node", "retrieve_node")
    graph.add_edge("reporter_credibility_node", "retrieve_node")

    graph.add_conditional_edges(
        "retrieve_node",
        _route_after_retrieve,
        {"score_node": "score_node", "urgency_node": "urgency_node"},
    )

    graph.add_edge("score_node", "hoax_node")
    graph.add_edge("hoax_node", "urgency_node")
    graph.add_edge("urgency_node", "final_score_node")
    graph.add_edge("final_score_node", "risk_assessment_node")
    graph.add_edge("risk_assessment_node", END)

    return graph


analysis_graph = build().compile()
