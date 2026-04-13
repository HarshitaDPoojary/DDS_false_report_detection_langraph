"""
evals/evaluators.py — LangSmith evaluator functions.

Each evaluator receives:
  run     — the LangSmith Run object (contains .outputs from the graph)
  example — the dataset Example object (contains .outputs = ground truth)

Returns a dict: {"key": str, "score": float, "comment": str}

Evaluators in this module:
  action_match          — did the graph recommend the right action?
  false_negative_guard  — real threat → not dismissed (safety-critical)
  false_positive_guard  — hoax → not escalated (resource waste)
  hoax_probability_direction — hoax → hp > 0.5; real → hp < 0.5
  confidence_calibration — is confidence_range appropriately wide?
  audit_trail_present   — did every node write to the audit trail?
"""
from __future__ import annotations

from typing import Any


def _get_outputs(run: Any) -> dict:
    """Extract graph outputs from a LangSmith Run."""
    outputs = run.outputs or {}
    # AnalysisGraph returns analysis_result as top-level key
    return outputs.get("analysis_result", outputs)


def _get_truth(example: Any) -> dict:
    return example.outputs or {}


# ---------------------------------------------------------------------------
# Evaluator 1 — Action match
# ---------------------------------------------------------------------------
def action_match(run: Any, example: Any) -> dict:
    """
    Score 1.0 if predicted action matches expected_action exactly.
    Score 0.5 if the predicted action is adjacent (e.g. human_review vs escalate).
    Score 0.0 for a complete mismatch.
    """
    outputs = _get_outputs(run)
    truth   = _get_truth(example)

    predicted = outputs.get("action", "monitor")
    expected  = truth.get("expected_action", "monitor")

    # Ordered severity: dismiss < monitor < human_review < escalate
    _order = {"dismiss": 0, "monitor": 1, "human_review": 2, "escalate": 3}
    p_rank = _order.get(predicted, 1)
    e_rank = _order.get(expected, 1)
    diff   = abs(p_rank - e_rank)

    if diff == 0:
        score, comment = 1.0, f"Exact match: {predicted}"
    elif diff == 1:
        score, comment = 0.5, f"Adjacent: predicted={predicted}, expected={expected}"
    else:
        score, comment = 0.0, f"Mismatch: predicted={predicted}, expected={expected}"

    return {"key": "action_match", "score": score, "comment": comment}


# ---------------------------------------------------------------------------
# Evaluator 2 — False negative guard (safety-critical)
# ---------------------------------------------------------------------------
def false_negative_guard(run: Any, example: Any) -> dict:
    """
    For real threats: score 0.0 if the system recommended 'dismiss'.
    Any other action on a real threat scores 1.0.
    Ambiguous cases are skipped (score=1.0 — no penalty).
    """
    outputs    = _get_outputs(run)
    truth      = _get_truth(example)
    true_label = truth.get("true_label", "ambiguous")

    if true_label != "real":
        return {"key": "false_negative_guard", "score": 1.0,
                "comment": f"Skipped — true_label={true_label}"}

    predicted = outputs.get("action", "monitor")
    if predicted == "dismiss":
        return {"key": "false_negative_guard", "score": 0.0,
                "comment": "CRITICAL: real threat dismissed"}

    return {"key": "false_negative_guard", "score": 1.0,
            "comment": f"OK — real threat action={predicted}"}


# ---------------------------------------------------------------------------
# Evaluator 3 — False positive guard
# ---------------------------------------------------------------------------
def false_positive_guard(run: Any, example: Any) -> dict:
    """
    For confirmed hoaxes: score 0.0 if the system escalated.
    Escalating a hoax wastes emergency resources.
    """
    outputs    = _get_outputs(run)
    truth      = _get_truth(example)
    true_label = truth.get("true_label", "ambiguous")

    if true_label != "hoax":
        return {"key": "false_positive_guard", "score": 1.0,
                "comment": f"Skipped — true_label={true_label}"}

    predicted = outputs.get("action", "monitor")
    if predicted == "escalate":
        return {"key": "false_positive_guard", "score": 0.0,
                "comment": "Hoax escalated — unnecessary resource dispatch"}

    return {"key": "false_positive_guard", "score": 1.0,
            "comment": f"OK — hoax action={predicted}"}


# ---------------------------------------------------------------------------
# Evaluator 4 — Hoax probability direction
# ---------------------------------------------------------------------------
def hoax_probability_direction(run: Any, example: Any) -> dict:
    """
    Confirms hoax_probability points in the right direction:
      real  → hoax_probability < 0.5  (score scaled from 0 to 1 as hp→0)
      hoax  → hoax_probability > 0.5  (score scaled from 0 to 1 as hp→1)
    Ambiguous cases: score 1.0 (no assertion).
    """
    outputs    = _get_outputs(run)
    truth      = _get_truth(example)
    true_label = truth.get("true_label", "ambiguous")
    hp         = float(outputs.get("hoax_probability", 0.5))

    if true_label == "ambiguous":
        return {"key": "hoax_probability_direction", "score": 1.0,
                "comment": "Skipped — ambiguous label"}

    if true_label == "real":
        # Perfect at hp=0; worst at hp=1
        score   = max(0.0, 1.0 - (hp / 0.5)) if hp <= 0.5 else 0.0
        comment = f"real: hoax_probability={hp:.3f}"
    else:  # hoax
        score   = max(0.0, (hp - 0.5) / 0.5) if hp >= 0.5 else 0.0
        comment = f"hoax: hoax_probability={hp:.3f}"

    return {"key": "hoax_probability_direction", "score": round(score, 3),
            "comment": comment}


# ---------------------------------------------------------------------------
# Evaluator 5 — Confidence calibration
# ---------------------------------------------------------------------------
def confidence_calibration(run: Any, example: Any) -> dict:
    """
    Checks that confidence_range:
      - is a 2-element list [low, high]
      - low < high
      - width is >= 0.08 (not overconfident)
      - width is <= 0.40 (not useless)
    Score 1.0 = all checks pass; deduct 0.25 per failure.
    """
    outputs = _get_outputs(run)
    cr      = outputs.get("confidence_range", [])
    deductions = 0
    comments   = []

    if not isinstance(cr, (list, tuple)) or len(cr) != 2:
        return {"key": "confidence_calibration", "score": 0.0,
                "comment": f"confidence_range malformed: {cr}"}

    low, high = float(cr[0]), float(cr[1])

    if low >= high:
        deductions += 1
        comments.append(f"low={low} >= high={high}")

    width = high - low
    if width < 0.08:
        deductions += 1
        comments.append(f"overconfident: width={width:.3f} < 0.08")
    if width > 0.40:
        deductions += 1
        comments.append(f"too wide: width={width:.3f} > 0.40")

    score   = max(0.0, 1.0 - deductions * 0.33)
    comment = "; ".join(comments) if comments else f"OK width={width:.3f}"

    return {"key": "confidence_calibration", "score": round(score, 2),
            "comment": comment}


# ---------------------------------------------------------------------------
# Evaluator 6 — Audit trail completeness
# ---------------------------------------------------------------------------
_REQUIRED_NODES = {
    "guardrails_node",
    "transform_node",
    "final_score_node",
}


def audit_trail_present(run: Any, example: Any) -> dict:
    """
    Verifies that core nodes wrote to the audit_trail.
    Score = fraction of required nodes present in audit_trail.
    """
    outputs     = run.outputs or {}
    audit_trail = outputs.get("audit_trail", [])
    nodes_seen  = {entry.get("node") for entry in audit_trail if isinstance(entry, dict)}

    missing = _REQUIRED_NODES - nodes_seen
    score   = 1.0 - (len(missing) / len(_REQUIRED_NODES))
    comment = (
        f"All required nodes present"
        if not missing
        else f"Missing: {', '.join(sorted(missing))}"
    )

    return {"key": "audit_trail_present", "score": round(score, 2),
            "comment": comment}


# ---------------------------------------------------------------------------
# All evaluators — import this list in run_evals.py
# ---------------------------------------------------------------------------
ALL_EVALUATORS = [
    action_match,
    false_negative_guard,
    false_positive_guard,
    hoax_probability_direction,
    confidence_calibration,
    audit_trail_present,
]
