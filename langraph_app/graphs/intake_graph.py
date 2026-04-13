"""
IntakeGraph — OCR + vision + EXIF extraction + LLM extraction pipeline.

Flow:
  guardrails_node          ← FIRST node; aborts on hard block
  check_attachments_node
    ├── (no files)   → extract_node  (directly)
    └── (has files)  → classify_attachments_node
                           ↓ (list-return fan-out — all relevant branches in parallel)
                           ├── ocr_node
                           ├── vision_node
                           ├── image_metadata_node
                           ├── screenshot_node     (only if has_screenshot)
                           ├── vehicle_node        (only if has_vehicle)
                           ├── id_document_node    (only if has_id_document)
                           └── person_node         (only if has_person)
                           all fan-in → extract_node
  extract_node → validate_node → END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from langraph_app.state import IntakeState
from langraph_app.nodes import guardrails, check_attachments, ocr, vision, image_metadata, extract, validate
from langraph_app.nodes import classify_attachments
from langraph_app.nodes import screenshot_node, vehicle_node, id_document_node, person_node


def _route_after_guardrails(state: IntakeState) -> str:
    return "abort" if state.get("guardrail_hard_block") else "continue"


def _route_after_check(state: IntakeState) -> str:
    return "has_attachments" if state.get("has_attachments") else "no_attachments"


def _route_after_classify(state: IntakeState) -> list[str]:
    """
    Fan-out to all relevant parallel branches.

    The three base nodes (ocr, vision, image_metadata) always run when
    attachments are present — they handle unknown/document types and EXIF.
    Specialized nodes are added conditionally based on what was classified.

    LangGraph v0.2+ supports list-return routing functions. extract_node
    waits for exactly the set of nodes returned here before running.
    """
    targets = ["ocr_node", "vision_node", "image_metadata_node"]
    if state.get("has_screenshot"):
        targets.append("screenshot_node")
    if state.get("has_vehicle"):
        targets.append("vehicle_node")
    if state.get("has_id_document"):
        targets.append("id_document_node")
    if state.get("has_person"):
        targets.append("person_node")
    return targets


def build() -> StateGraph:
    graph = StateGraph(IntakeState)

    # ── Register all nodes ───────────────────────────────────────────────────
    graph.add_node("guardrails_node",            guardrails.run)
    graph.add_node("check_attachments_node",     check_attachments.run)
    graph.add_node("classify_attachments_node",  classify_attachments.run)
    graph.add_node("ocr_node",                   ocr.run)
    graph.add_node("vision_node",                vision.run)
    graph.add_node("image_metadata_node",        image_metadata.run)
    graph.add_node("screenshot_node",            screenshot_node.run)
    graph.add_node("vehicle_node",               vehicle_node.run)
    graph.add_node("id_document_node",           id_document_node.run)
    graph.add_node("person_node",                person_node.run)
    graph.add_node("extract_node",               extract.run)
    graph.add_node("validate_node",              validate.run)

    # ── Entry point ──────────────────────────────────────────────────────────
    graph.set_entry_point("guardrails_node")

    # guardrails → abort or continue
    graph.add_conditional_edges(
        "guardrails_node",
        _route_after_guardrails,
        {"abort": END, "continue": "check_attachments_node"},
    )

    # check_attachments → no files: go straight to extract
    #                   → has files: classify first
    graph.add_conditional_edges(
        "check_attachments_node",
        _route_after_check,
        {
            "no_attachments": "extract_node",
            "has_attachments": "classify_attachments_node",
        },
    )

    # classify_attachments → parallel fan-out (list-return routing)
    graph.add_conditional_edges(
        "classify_attachments_node",
        _route_after_classify,
    )

    # Fan-in: all parallel branches → extract_node
    for node_name in (
        "ocr_node",
        "vision_node",
        "image_metadata_node",
        "screenshot_node",
        "vehicle_node",
        "id_document_node",
        "person_node",
    ):
        graph.add_edge(node_name, "extract_node")

    graph.add_edge("extract_node", "validate_node")
    graph.add_edge("validate_node", END)

    return graph


intake_graph = build().compile()
