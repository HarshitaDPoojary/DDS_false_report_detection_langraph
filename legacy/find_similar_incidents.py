"""
Find Similar Incidents - Main Search Script

Multi-dimensional similarity search for incident reports using:
- Geographic proximity (geo-spatial filtering)
- Temporal proximity (time window filtering)
- Semantic text similarity (kNN on embeddings)
- Incident type matching

Usage:
    python find_similar_incidents.py --query sample_report.json --radius 5 --lookback-hours 24
    python find_similar_incidents.py --query sample_report.json --radius 10 --lookback-days 7 --verbose
"""

import json
import argparse
from elasticsearch import Elasticsearch
from env_loader import load_es_config
from es_ingest_data import transform_report, validate_document, generate_text_embedding
from es_query_builder import build_combined_query
from similarity_scoring import SimilarityScorer


def load_query_report(file_path):
    """
    Load and transform query report from JSON file.

    Args:
        file_path: Path to JSON file containing report

    Returns:
        Transformed query report dict
    """
    with open(file_path) as f:
        raw_report = json.load(f)

    # Transform and validate
    transformed = transform_report(raw_report)
    validate_document(transformed)

    return transformed


def execute_similarity_search(
    query_report,
    radius_miles=5.0,
    lookback_hours=24,
    lookback_days=None,
    limit=10,
    scorer_weights=None
):
    """
    Execute multi-dimensional similarity search.

    Args:
        query_report: Transformed query report dict
        radius_miles: Geo search radius in miles (default: 5.0)
        lookback_hours: Time lookback in hours (default: 24)
        lookback_days: Time lookback in days (overrides lookback_hours if set)
        limit: Maximum number of results (default: 10)
        scorer_weights: Optional dict with custom scorer weights
                       e.g., {"geo_weight": 0.3, "time_weight": 0.2}

    Returns:
        List of ranked, scored results
    """
    # Load ES config
    config = load_es_config()
    es = Elasticsearch(config["host"], api_key=config["api_key"])

    # Extract query parameters
    location = query_report.get("location", [])
    lon, lat = location[0], location[1]

    time_start = query_report.get("time_start")
    time_end = query_report.get("time_end")

    query_embedding = query_report.get("text_embedding", [])
    incident_type = query_report.get("incident_type", "")
    report_id = query_report.get("report_id")

    # Build combined query
    query = build_combined_query(
        lat=lat,
        lon=lon,
        start=time_start,
        end=time_end,
        query_embedding=query_embedding,
        incident_type=incident_type,
        radius_miles=radius_miles,
        lookback_hours=lookback_hours,
        lookback_days=lookback_days,
        exclude_report_id=report_id,
        size=limit * 2  # Get more results for re-ranking
    )

    # Execute search
    print(f"Searching for similar incidents...")
    print(f"  Location: ({lat:.4f}, {lon:.4f})")
    print(f"  Radius: {radius_miles} miles")
    if lookback_days:
        print(f"  Time window: {lookback_days} days")
    else:
        print(f"  Time window: {lookback_hours} hours")
    print(f"  Incident type: {incident_type}")
    print()

    response = es.search(index=config["index"], body=query)
    hits = response["hits"]["hits"]

    print(f"Found {len(hits)} matching incidents from Elasticsearch")

    if not hits:
        return []

    # Initialize scorer with custom weights if provided
    if scorer_weights:
        scorer = SimilarityScorer(**scorer_weights)
    else:
        scorer = SimilarityScorer()

    # Score and rank results
    print(f"Ranking results by multi-dimensional similarity...")
    scored_results = scorer.rank_results(query_report, hits)

    # Limit to requested number
    return scored_results[:limit]


def format_results(scored_results, verbose=False):
    """
    Format scored results for display.

    Args:
        scored_results: List of scored result dicts
        verbose: Show detailed score breakdown (default: False)

    Returns:
        List of formatted result strings
    """
    formatted = []

    for i, result in enumerate(scored_results, 1):
        final_score = result["final_score"]
        geo_info = result["geo"]
        time_info = result["time"]
        text_info = result["text"]
        type_info = result["type"]
        source = result["result"]

        # Header
        lines = [
            f"\n{'='*80}",
            f"Rank #{i} - Overall Similarity: {final_score:.3f}",
            f"{'='*80}"
        ]

        # Basic info
        lines.append(f"Report ID: {source.get('report_id', 'N/A')}")
        lines.append(f"Incident Type: {source.get('incident_type', 'N/A')}")
        lines.append(f"Location: {source.get('where', {}).get('venue', 'N/A')}")

        # Similarity scores (summary)
        lines.append(f"\nSimilarity Scores:")
        lines.append(f"  Geographic: {geo_info['score']:.3f} ({geo_info['explanation']})")
        lines.append(f"  Temporal:   {time_info['score']:.3f} ({time_info['explanation']})")
        lines.append(f"  Text:       {text_info['score']:.3f} ({text_info['explanation']})")
        lines.append(f"  Type:       {type_info['score']:.3f} ({type_info['explanation']})")

        # Text preview
        free_text = source.get("free_text", "")
        preview = free_text[:200] + "..." if len(free_text) > 200 else free_text
        lines.append(f"\nIncident Description:")
        lines.append(f"  {preview}")

        # Verbose details
        if verbose:
            lines.append(f"\nDetailed Information:")
            lines.append(f"  Time Window: {source.get('time_start', 'N/A')} to {source.get('time_end', 'N/A')}")
            lines.append(f"  Location Coords: {source.get('location', 'N/A')}")
            lines.append(f"  Duration: {source.get('time_duration_hours', 0):.1f} hours")

            classification = source.get("incident_classification", {})
            if classification:
                lines.append(f"  Severity: {classification.get('severity', 'N/A')}")
                lines.append(f"  Urgency: {classification.get('urgency_score', 0):.2f}")

                classified_types = classification.get("types", [])
                if classified_types:
                    lines.append(f"  Classified Types:")
                    for ct in classified_types[:3]:  # Show top 3
                        if isinstance(ct, dict):
                            lines.append(f"    - {ct.get('type', 'N/A')} (confidence: {ct.get('confidence', 0):.2f})")

        formatted.append("\n".join(lines))

    return formatted


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Find similar incidents using multi-dimensional similarity search"
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Path to query report JSON file (e.g., sample_report.json)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Geographic search radius in miles (default: 5.0)"
    )
    parser.add_argument(
        "--lookback-hours",
        type=float,
        default=24,
        help="Time lookback window in hours (default: 24)"
    )
    parser.add_argument(
        "--lookback-days",
        type=float,
        help="Time lookback window in days (overrides --lookback-hours)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results to return (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed score breakdown and metadata"
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Load query report
    print(f"Loading query report from {args.query}...")
    query_report = load_query_report(args.query)
    print(f"Query report loaded: {query_report.get('report_id', 'N/A')}")
    print()

    # Execute search
    results = execute_similarity_search(
        query_report,
        radius_miles=args.radius,
        lookback_hours=args.lookback_hours,
        lookback_days=args.lookback_days,
        limit=args.limit
    )

    # Display results
    print()
    if not results:
        print("No similar incidents found.")
    else:
        print(f"\nTop {len(results)} Similar Incidents:")
        print("="*80)

        formatted = format_results(results, verbose=args.verbose)
        for result_str in formatted:
            print(result_str)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
