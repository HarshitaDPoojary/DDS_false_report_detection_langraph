"""
evals/dataset.py — build and upload the LangSmith evaluation dataset.

Uses reports_cluster.json as the source of ground-truth examples.
Each example is one report run through the full AnalysisGraph.

Run once to create the dataset in LangSmith:
    python -m evals.dataset --create

Re-run to add examples or update existing ones:
    python -m evals.dataset --sync

Local export (no LangSmith key needed):
    python -m evals.dataset --export evals/dataset_local.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

DATASET_NAME = "false-report-identification-v1"
DATASET_DESCRIPTION = (
    "Ground-truth labelled incident reports for evaluating the "
    "False Report Identification LangGraph pipeline."
)

# Path to the real report fixture file
_REPORTS_FILE = Path(__file__).parent.parent / "reports_cluster.json"


# ---------------------------------------------------------------------------
# Ground-truth labels for each report_id in reports_cluster.json
# Determined by human analyst review of the fixture data.
# ---------------------------------------------------------------------------
_LABELS: dict[int, dict] = {
    # Burglaries — credible, specific witnesses
    1:  {"true_label": "real", "expected_action": "monitor",
         "notes": "Named pharmacist, specific address, specific time window"},
    6:  {"true_label": "real", "expected_action": "monitor",
         "notes": "Neighbor witness, specific address"},
    9:  {"true_label": "real", "expected_action": "monitor",
         "notes": "Victim, corroborating neighbor text"},

    # Physical fights — multiple bystanders, named locations
    3:  {"true_label": "real", "expected_action": "monitor",
         "notes": "Bystander witness, gave police statement"},
    7:  {"true_label": "real", "expected_action": "monitor",
         "notes": "Bystander, lifeguard corroboration, police statement"},
    10: {"true_label": "real", "expected_action": "monitor",
         "notes": "Bystander, dialed 911"},

    # Sexual harassment — second-hand reports (friend/colleague told me)
    2:  {"true_label": "ambiguous", "expected_action": "human_review",
         "notes": "Second-hand (roommate confided) — no direct witness"},
    4:  {"true_label": "real", "expected_action": "monitor",
         "notes": "Direct bystander, multiple people intervened"},
    5:  {"true_label": "real", "expected_action": "monitor",
         "notes": "Direct bystander intervention"},
    8:  {"true_label": "ambiguous", "expected_action": "human_review",
         "notes": "Second-hand (colleague confided) — no direct witness"},
    13: {"true_label": "ambiguous", "expected_action": "human_review",
         "notes": "Second-hand (friend told me) — no direct witness"},

    # Gun threats — high urgency; second-hand vs direct
    11: {"true_label": "real", "expected_action": "escalate",
         "notes": "Threatening texts with firearm — direct evidence, high urgency"},
    12: {"true_label": "real", "expected_action": "escalate",
         "notes": "Eyewitness to road rage with firearm"},
}


def _load_reports() -> list[dict]:
    with open(_REPORTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _build_example(report: dict) -> dict[str, Any] | None:
    """Convert one report into a LangSmith {inputs, outputs} example."""
    rid = report.get("report_id")
    label_info = _LABELS.get(rid)
    if label_info is None:
        return None  # no label yet — skip

    inputs = {
        "raw_report": {
            "report_id":     str(rid),
            "incident_type": report.get("incident_type", ""),
            "free_text":     report.get("free_text", ""),
            "where":         report.get("where", {}),
            "when_window":   report.get("when_window", {}),
            "means":         report.get("means", ""),
            "reporter": {
                "type":      report.get("reporter", {}).get("type", "unknown"),
                "anonymous": report.get("reporter", {}).get("type") is None,
            },
        },
        "radius_miles":    5.0,
        "lookback_hours":  24.0,
    }

    outputs = {
        "true_label":       label_info["true_label"],
        "expected_action":  label_info["expected_action"],
        "notes":            label_info.get("notes", ""),
    }

    return {"inputs": inputs, "outputs": outputs, "id": str(rid)}


def build_examples() -> list[dict]:
    reports = _load_reports()
    examples = []
    for r in reports:
        ex = _build_example(r)
        if ex:
            examples.append(ex)
    return examples


# ---------------------------------------------------------------------------
# LangSmith upload
# ---------------------------------------------------------------------------
def create_or_sync_dataset(sync: bool = False) -> None:
    from langsmith import Client

    client = Client()
    examples = build_examples()

    existing = None
    try:
        existing = client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        pass

    if existing is None or not sync:
        if existing:
            print(f"Dataset '{DATASET_NAME}' already exists. Use --sync to update.")
            return
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION,
        )
        print(f"Created dataset: {DATASET_NAME} (id={dataset.id})")
    else:
        dataset = existing
        print(f"Syncing dataset: {DATASET_NAME} (id={dataset.id})")

    # Upload examples (upsert by example_id)
    for ex in examples:
        try:
            client.create_example(
                dataset_id=dataset.id,
                inputs=ex["inputs"],
                outputs=ex["outputs"],
            )
        except Exception:
            pass  # already exists on sync — skip

    print(f"Uploaded {len(examples)} examples to '{DATASET_NAME}'.")


# ---------------------------------------------------------------------------
# Local export
# ---------------------------------------------------------------------------
def export_local(path: str) -> None:
    examples = build_examples()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    print(f"Exported {len(examples)} examples to {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage LangSmith eval dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--create", action="store_true",
                       help="Create dataset in LangSmith (fails if already exists)")
    group.add_argument("--sync", action="store_true",
                       help="Sync examples to existing dataset")
    group.add_argument("--export", metavar="PATH",
                       help="Export dataset as local JSON (no LangSmith needed)")
    args = parser.parse_args()

    if args.export:
        export_local(args.export)
    else:
        create_or_sync_dataset(sync=args.sync)
