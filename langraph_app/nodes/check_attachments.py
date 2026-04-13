"""
check_attachments_node — routing node for IntakeGraph.

Checks whether the submission includes any file attachments.
The graph uses this to decide whether to fan out to ocr_node /
vision_node / image_metadata_node, or go straight to extract_node.
"""
from __future__ import annotations


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    attachments = state.get("attachments", [])
    return {"has_attachments": len(attachments) > 0}
