"""
Process reports from reports.json and calculate urgency scores

This script:
1. Loads reports from reports.json
2. Classifies each free_text using multiple LLM providers (OpenAI, Claude, Local)
3. Aggregates incident types from all providers
4. Calculates urgency score based on consensus incident type and contextual factors
5. Outputs results with detailed breakdown
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from collections import Counter

from incident_severity_score import (
    get_incident_types,
    call_openai,
    call_claude,
    call_local_transformers,
    build_llm_prompt_for_types,
    BASE_SCORE_BY_TYPE,
    DEFAULT_LLM_MODEL,
    DEFAULT_CLAUDE_MODEL,
)
from severity_urgency_score import calculate_urgency_score, get_urgency_level


# -----------------------------------------
# Multi-provider incident classification
# -----------------------------------------

def classify_with_multiple_providers(
    text: str,
    providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Classify text using multiple LLM providers and aggregate results.
    
    Args:
        text: The incident report text
        providers: List of providers to use ["local", "openai", "claude"]
                  If None, tries all available based on env vars
    
    Returns:
        {
            "consensus_type": str,
            "consensus_confidence": float,
            "base_score": float,
            "providers_used": [str],
            "all_results": {provider: result},
            "vote_breakdown": {type: count}
        }
    """
    if providers is None:
        # Determine which providers to use based on env
        providers = ["local"]  # Always include local (rule-based)
        if os.environ.get("OPENAI_API_KEY"):
            providers.append("openai")
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"):
            providers.append("claude")
    
    all_results = {}
    all_types = []
    
    # Get local rule-based classification first
    if "local" in providers:
        try:
            local_result = get_incident_types(text, top_k=1, min_score=0.0)
            if local_result:
                all_results["local"] = local_result[0]
                all_types.append(local_result[0]["type"])
        except Exception as e:
            print(f"  [local classifier failed: {e}]")
    
    # Build LLM prompt once
    prompt = build_llm_prompt_for_types(text)
    
    # Try OpenAI
    if "openai" in providers:
        try:
            openai_model = os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)
            response = call_openai(prompt, model=openai_model)
            parsed = json.loads(response) if isinstance(response, str) else response
            if parsed and "types" in parsed and parsed["types"]:
                openai_result = parsed["types"][0]
                all_results["openai"] = openai_result
                all_types.append(openai_result["type"])
        except Exception as e:
            print(f"  [OpenAI failed: {e}]")
    
    # Try Claude
    if "claude" in providers:
        try:
            claude_model = os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
            response = call_claude(prompt, model=claude_model)
            parsed = json.loads(response) if isinstance(response, str) else response
            if parsed and "types" in parsed and parsed["types"]:
                claude_result = parsed["types"][0]
                all_results["claude"] = claude_result
                all_types.append(claude_result["type"])
        except Exception as e:
            print(f"  [Claude failed: {e}]")
    
    # Aggregate results
    if not all_types:
        # Fallback if all providers failed
        return {
            "consensus_type": "other",
            "consensus_confidence": 0.1,
            "base_score": BASE_SCORE_BY_TYPE.get("other", 1.0),
            "providers_used": [],
            "all_results": {},
            "vote_breakdown": {"other": 0}
        }
    
    # Vote counting
    vote_counts = Counter(all_types)
    most_common = vote_counts.most_common(1)[0]
    consensus_type = most_common[0]
    consensus_count = most_common[1]
    
    # Calculate consensus confidence (simple average of matching types)
    matching_confidences = []
    for provider, result in all_results.items():
        if result.get("type") == consensus_type:
            matching_confidences.append(result.get("confidence", 0.5))
    
    consensus_confidence = sum(matching_confidences) / len(matching_confidences) if matching_confidences else 0.5
    
    # If there's disagreement, average the base scores
    if len(vote_counts) > 1:
        # Multiple different types detected - average their base scores
        unique_types = list(vote_counts.keys())
        base_scores = [BASE_SCORE_BY_TYPE.get(t, 1.0) for t in unique_types]
        base_score = sum(base_scores) / len(base_scores)
    else:
        # All agree on same type
        base_score = BASE_SCORE_BY_TYPE.get(consensus_type, 1.0)
    
    return {
        "consensus_type": consensus_type,
        "consensus_confidence": round(consensus_confidence, 3),
        "base_score": round(base_score, 2),
        "providers_used": list(all_results.keys()),
        "all_results": all_results,
        "vote_breakdown": dict(vote_counts)
    }


# -----------------------------------------
# Report processing
# -----------------------------------------

def process_single_report(report: Dict[str, Any], providers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Process a single report and return classification + urgency."""
    report_id = report.get("report_id", "unknown")
    free_text = report.get("free_text", "")
    
    if not free_text:
        return {
            "report_id": report_id,
            "error": "No free_text field",
        }
    
    print(f"\nProcessing: {report_id}")
    print(f"Text: {free_text[:100]}...")
    
    # Step 1: Multi-provider classification
    classification = classify_with_multiple_providers(free_text, providers=providers)
    
    print(f"  Consensus: {classification['consensus_type']} (confidence: {classification['consensus_confidence']})")
    print(f"  Providers: {', '.join(classification['providers_used'])}")
    if len(classification['vote_breakdown']) > 1:
        print(f"  Votes: {classification['vote_breakdown']}")
    
    # Step 2: Calculate urgency using consensus type
    # Create a mock incident_types list for urgency calculation
    mock_incident_types = [{
        "type": classification["consensus_type"],
        "confidence": classification["consensus_confidence"],
        "matches": ["consensus from providers"]
    }]
    
    urgency_result = calculate_urgency_score(
        free_text,
        incident_types=mock_incident_types,
        normalize=True
    )
    
    urgency_level = get_urgency_level(urgency_result["score"])
    
    print(f"  Urgency: {urgency_result['score']:.3f} ({urgency_level})")
    
    # Combine results
    return {
        "report_id": report_id,
        "original_incident_type": report.get("incident_type"),
        "classification": classification,
        "urgency": urgency_result,
        "urgency_level": urgency_level,
    }


def process_reports_file(
    filepath: str = "reports.json",
    providers: Optional[List[str]] = None,
    output_file: Optional[str] = "processed_reports.json",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Process all reports from a JSON file.
    
    Args:
        filepath: Path to reports.json
        providers: List of providers to use (None = auto-detect)
        output_file: Where to save results (None = don't save)
        limit: Max number of reports to process (None = all)
    
    Returns:
        List of processed report results
    """
    # Load reports
    with open(filepath, 'r', encoding='utf-8') as f:
        reports = json.load(f)
    
    print(f"Loaded {len(reports)} reports from {filepath}")
    
    if limit:
        reports = reports[:limit]
        print(f"Processing first {limit} reports...")
    
    # Process each report
    results = []
    for i, report in enumerate(reports, 1):
        print(f"\n{'='*80}")
        print(f"Report {i}/{len(reports)}")
        print(f"{'='*80}")
        
        try:
            result = process_single_report(report, providers=providers)
            results.append(result)
        except Exception as e:
            print(f"ERROR processing {report.get('report_id')}: {e}")
            results.append({
                "report_id": report.get("report_id", "unknown"),
                "error": str(e)
            })
    
    # Save results if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if "error" not in r]
    print(f"Successfully processed: {len(successful)}/{len(results)}")
    
    if successful:
        # Urgency distribution
        urgency_levels = Counter([r["urgency_level"] for r in successful])
        print("\nUrgency Distribution:")
        for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]:
            count = urgency_levels.get(level, 0)
            if count > 0:
                print(f"  {level}: {count}")
        
        # Average urgency by original incident type
        by_type = {}
        for r in successful:
            orig_type = r.get("original_incident_type", "unknown")
            if orig_type not in by_type:
                by_type[orig_type] = []
            by_type[orig_type].append(r["urgency"]["score"])
        
        print("\nAverage Urgency by Original Incident Type:")
        for itype, scores in sorted(by_type.items()):
            avg_score = sum(scores) / len(scores)
            print(f"  {itype}: {avg_score:.3f} (n={len(scores)})")
    
    return results


# -----------------------------------------
# CLI entry point
# -----------------------------------------

def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process incident reports with multi-provider classification and urgency scoring")
    parser.add_argument("--file", default="reports.json", help="Path to reports JSON file")
    parser.add_argument("--output", default="processed_reports.json", help="Output file for results")
    parser.add_argument("--limit", type=int, help="Limit number of reports to process")
    parser.add_argument("--providers", nargs="+", choices=["local", "openai", "claude"], 
                       help="Specific providers to use (default: auto-detect)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    
    args = parser.parse_args()
    
    output_file = None if args.no_save else args.output
    
    results = process_reports_file(
        filepath=args.file,
        providers=args.providers,
        output_file=output_file,
        limit=args.limit,
    )
    
    return results


if __name__ == "__main__":
    # Example: process first 5 reports for demo
    results = process_reports_file(
        filepath="reports.json",
        output_file="processed_reports.json",
        limit=5,  # Remove or set to None to process all
    )
