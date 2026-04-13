"""
evals/run_evals.py — run the full evaluation suite.

Usage:
    # Against LangSmith dataset (requires LANGCHAIN_API_KEY env var)
    python -m evals.run_evals

    # Local mode (no LangSmith account needed — uses exported JSON)
    python -m evals.run_evals --local evals/dataset_local.json

    # Filter to a tag subset
    python -m evals.run_evals --local evals/dataset_local.json --tag gun_threat

    # Dry run — print cases without invoking the graph
    python -m evals.run_evals --local evals/dataset_local.json --dry-run

Environment variables:
    LANGCHAIN_TRACING_V2=true       — enables auto-tracing of every graph run
    LANGCHAIN_API_KEY=ls__...       — LangSmith API key
    LANGCHAIN_PROJECT=false-report-identification
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from evals.evaluators import ALL_EVALUATORS
from evals.dataset import DATASET_NAME

# ---------------------------------------------------------------------------
# Local mode helpers
# ---------------------------------------------------------------------------
class _LocalRun:
    """Mimics a LangSmith Run object for the evaluator interface."""
    def __init__(self, outputs: dict):
        self.outputs = outputs


class _LocalExample:
    """Mimics a LangSmith Example object for the evaluator interface."""
    def __init__(self, outputs: dict, metadata: dict | None = None):
        self.outputs = outputs
        self.metadata = metadata or {}


def _run_local(dataset_path: str, tag_filter: str | None, dry_run: bool) -> None:
    with open(dataset_path, encoding="utf-8") as f:
        examples = json.load(f)

    from langraph_app.graphs.analysis_graph import analysis_graph

    results_by_evaluator: dict[str, list[float]] = {
        e.__name__: [] for e in ALL_EVALUATORS
    }
    total = passed = 0
    fn_failures = []  # false-negative safety failures

    for ex in examples:
        inputs  = ex["inputs"]
        outputs = ex["outputs"]

        # Tag filter
        if tag_filter:
            label = outputs.get("expected_action", "")
            true_label = outputs.get("true_label", "")
            tags = [label, true_label]
            if tag_filter not in tags:
                continue

        total += 1

        if dry_run:
            print(f"[DRY RUN] {inputs['raw_report']['report_id']} "
                  f"— expected={outputs['expected_action']} ({outputs['true_label']})")
            continue

        # Run the graph
        try:
            state_out = analysis_graph.invoke(inputs)
        except Exception as exc:
            print(f"  ERROR invoking graph for report "
                  f"{inputs['raw_report'].get('report_id')}: {exc}")
            continue

        run     = _LocalRun(state_out)
        example = _LocalExample(outputs)

        # Run all evaluators
        case_passed = True
        print(f"\nReport {inputs['raw_report']['report_id']} "
              f"— true={outputs['true_label']} expected={outputs['expected_action']}")

        for evaluator in ALL_EVALUATORS:
            result = evaluator(run, example)
            score  = result["score"]
            key    = result["key"]
            comment = result.get("comment", "")
            results_by_evaluator[key].append(score)

            status = "PASS" if score >= 0.5 else "FAIL"
            if score < 0.5:
                case_passed = False
            if key == "false_negative_guard" and score == 0.0:
                fn_failures.append(
                    f"report_id={inputs['raw_report'].get('report_id')} — {comment}"
                )
            print(f"  {status} [{key}] score={score:.2f}  {comment}")

        if case_passed:
            passed += 1

    if dry_run:
        print(f"\n{total} cases would be evaluated.")
        return

    # Summary
    print("\n" + "=" * 52)
    print("  EVALUATION SUMMARY")
    print("=" * 52)
    print(f"  Cases run:   {total}")
    print(f"  Cases passed: {passed}  ({passed/max(total,1)*100:.1f}%)")
    print("-" * 52)
    for key, scores in results_by_evaluator.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {key:<32} avg={avg:.3f}")
    print("-" * 52)
    if fn_failures:
        print(f"  ⚠  FALSE NEGATIVE FAILURES ({len(fn_failures)}):")
        for f in fn_failures:
            print(f"    • {f}")
    else:
        print("  ✓  No false-negative failures (real threats not dismissed)")
    print("=" * 52)


# ---------------------------------------------------------------------------
# LangSmith mode
# ---------------------------------------------------------------------------
def _run_langsmith(experiment_prefix: str | None) -> None:
    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()

    # Verify dataset exists
    try:
        client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        print(
            f"Dataset '{DATASET_NAME}' not found in LangSmith.\n"
            "Run: python -m evals.dataset --create"
        )
        sys.exit(1)

    from langraph_app.graphs.analysis_graph import analysis_graph

    def _graph_fn(inputs: dict) -> dict:
        return analysis_graph.invoke(inputs)

    prefix = experiment_prefix or "eval"
    experiment_name = f"{prefix}-{int(time.time())}"

    print(f"Running experiment '{experiment_name}' against dataset '{DATASET_NAME}'…")

    results = evaluate(
        _graph_fn,
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=prefix,
        metadata={"graph_version": "analysis_graph_v1"},
    )

    print(f"\nResults stored in LangSmith project.")
    print(f"Experiment URL: {results.experiment_name}")

    # Print aggregate scores locally as well
    agg: dict[str, list[float]] = {}
    for r in results:
        for fb in (r.feedback or []):
            agg.setdefault(fb.key, []).append(fb.score or 0.0)

    print("\nAggregate scores:")
    for key, scores in agg.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  {key:<32} avg={avg:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run evaluation suite for the False Report Identification graph"
    )
    parser.add_argument(
        "--local", metavar="DATASET_JSON",
        help="Run in local mode using exported JSON dataset (no LangSmith account needed)",
    )
    parser.add_argument(
        "--tag", metavar="TAG",
        help="Only run cases where expected_action or true_label matches TAG",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print cases without invoking the graph",
    )
    parser.add_argument(
        "--experiment", metavar="PREFIX", default=None,
        help="LangSmith experiment name prefix (default: 'eval')",
    )
    args = parser.parse_args()

    if args.local:
        _run_local(args.local, tag_filter=args.tag, dry_run=args.dry_run)
    else:
        _run_langsmith(experiment_prefix=args.experiment)
