"""
CLI entry point for the False Report Identification system.

Usage:
    python -m langraph_app.cli --query sample_report.json
    python -m langraph_app.cli --query sample_report.json --radius 10 --lookback-hours 6
    python -m langraph_app.cli --text "Bomb threat at Lincoln High School" --radius 5

Options:
    --query PATH           JSON file containing a raw report dict
    --text TEXT            Report text (alternative to --query for quick tests)
    --radius FLOAT         Search radius in miles (default: 5.0)
    --lookback-hours FLOAT Lookback window in hours (default: 24.0)
    --json                 Output full JSON result instead of summary
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from langraph_app.graphs.analysis_graph import analysis_graph


def _build_state(raw_report: dict, radius: float, lookback: float) -> dict:
    report_id = raw_report.get("report_id") or str(uuid.uuid4())
    free_text = (
        raw_report.get("free_text")
        or raw_report.get("text_input")
        or raw_report.get("report", "")
    )
    return {
        "report_id": report_id,
        "raw_report": raw_report,
        "free_text": free_text,
        "location": raw_report.get("location", []),
        "time_start": raw_report.get("time_start", ""),
        "time_end": raw_report.get("time_end", ""),
        "time_midpoint": raw_report.get("time_midpoint", ""),
        "text_embedding": raw_report.get("text_embedding", []),
        "incident_types": raw_report.get("incident_types", []),
        "severity": raw_report.get("severity", "low"),
        "image_metadata": [],
        "image_metadata_conflicts": [],
        "visual_description": "",
        "soc_hash": raw_report.get("soc_hash", ""),
        "extraction_result": {},
        "radius_miles": radius,
        "lookback_hours": lookback,
        "scorer_weights": {},
    }


def _print_summary(result: dict) -> None:
    hp = result.get("hoax_probability", 0.0)
    tl = result.get("threat_level", 0.0)
    action = result.get("action", "unknown")
    cr = result.get("confidence_range", [hp, hp])
    urgency = result.get("urgency_level", "MINIMAL")
    cluster = result.get("cluster_summary", {})
    risk = result.get("false_negative_risk", "low")
    report_id = result.get("report_id", "")

    print(f"\n{'═'*60}")
    print(f"  Report ID     : {report_id}")
    print(f"  Hoax Prob     : {hp:.2f}  [{cr[0]:.2f} – {cr[1]:.2f}]")
    print(f"  Threat Level  : {tl:.2f}  ({urgency})")
    print(f"  Action        : {action.upper()}")
    print(f"  FN Risk       : {risk.upper()}")
    print(f"  Cluster Size  : {cluster.get('cluster_size', 0)}")
    print(f"{'─'*60}")

    ai_analysis = result.get("ai_analysis", "")
    if ai_analysis:
        print("  AI Analysis:")
        for line in ai_analysis.split("\n"):
            print(f"    {line}")

    top_results = (result.get("scored_results") or [])[:3]
    if top_results:
        print(f"\n  Top {len(top_results)} similar report(s):")
        for r in top_results:
            print(f"    [{r.get('final_score', 0):.2f}] {r.get('report_id', '?')}")

    print(f"{'═'*60}\n")


async def _run(args: argparse.Namespace) -> int:
    if args.query:
        with open(args.query, encoding="utf-8") as f:
            raw_report = json.load(f)
    elif args.text:
        raw_report = {"report_id": str(uuid.uuid4()), "free_text": args.text}
    else:
        print("Error: provide --query or --text", file=sys.stderr)
        return 1

    state = _build_state(raw_report, radius=args.radius, lookback=args.lookback_hours)

    print(f"Running analysis for report {state['report_id']} …")
    result = await analysis_graph.ainvoke(state)

    if result.get("guardrail_hard_block"):
        print(f"Guardrail block: {result.get('error', 'unknown')}", file=sys.stderr)
        return 2

    if result.get("error"):
        print(f"Error: {result['error']}", file=sys.stderr)
        return 2

    analysis = result.get("analysis_result") or result
    analysis["false_negative_risk"] = result.get("false_negative_risk", "low")
    analysis["escalation_reason"] = result.get("escalation_reason", "")

    if args.json:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        _print_summary(analysis)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="False Report Identification — CLI runner"
    )
    parser.add_argument("--query", metavar="PATH", help="JSON report file")
    parser.add_argument("--text", metavar="TEXT", help="Report text (quick test)")
    parser.add_argument("--radius", type=float, default=5.0, help="Search radius in miles")
    parser.add_argument("--lookback-hours", type=float, default=24.0, dest="lookback_hours")
    parser.add_argument("--json", action="store_true", help="Output full JSON result")
    args = parser.parse_args()

    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
