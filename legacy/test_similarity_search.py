"""
Test Suite for Similarity Search System

Tests the multi-dimensional similarity search implementation:
- Data loading and transformation
- Elasticsearch indexing
- Similarity search queries
- Score normalization
- Edge cases
"""

import json
from es_ingest_data import (
    extract_location,
    extract_time_window,
    classify_incident,
    transform_report,
    validate_document
)
from find_similar_incidents import (
    load_query_report,
    execute_similarity_search
)
from similarity_scoring import SimilarityScorer


def test_extract_location():
    """Test location extraction from different formats."""
    print("\n" + "="*80)
    print("TEST: Extract Location")
    print("="*80)

    # Format 1: coordinates.{lat, lon}
    report1 = {
        "where": {
            "coordinates": {"lat": 37.7749, "lon": -122.4194}
        }
    }
    location1 = extract_location(report1)
    print(f"Format 1 (coordinates): {location1}")
    assert location1 == [-122.4194, 37.7749], "Format 1 failed"

    # Format 2: geo array [lat, lon]
    report2 = {
        "where": {
            "geo": [37.7749, -122.4194]
        }
    }
    location2 = extract_location(report2)
    print(f"Format 2 (geo array): {location2}")
    assert location2 == [-122.4194, 37.7749], "Format 2 failed"

    # Format 3: geo_point GeoJSON
    report3 = {
        "where": {
            "geo_point": {
                "type": "Point",
                "coordinates": [-122.4194, 37.7749]
            }
        }
    }
    location3 = extract_location(report3)
    print(f"Format 3 (geo_point): {location3}")
    assert location3 == [-122.4194, 37.7749], "Format 3 failed"

    print("✓ All location extraction tests passed")


def test_extract_time_window():
    """Test time window extraction from different formats."""
    print("\n" + "="*80)
    print("TEST: Extract Time Window")
    print("="*80)

    # Format 1: earliest/latest
    report1 = {
        "when_window": {
            "earliest": "2025-09-27T02:30:00Z",
            "latest": "2025-09-27T03:00:00Z"
        }
    }
    time1 = extract_time_window(report1)
    print(f"Format 1 (earliest/latest):")
    print(f"  Start: {time1['start']}")
    print(f"  End: {time1['end']}")
    print(f"  Duration: {time1['duration_hours']:.1f} hours")
    assert time1["duration_hours"] == 0.5, "Format 1 duration failed"

    # Format 2: start_iso/end_iso
    report2 = {
        "when_window": {
            "start_iso": "2025-09-27T02:30:00Z",
            "end_iso": "2025-09-27T03:00:00Z"
        }
    }
    time2 = extract_time_window(report2)
    print(f"Format 2 (start_iso/end_iso):")
    print(f"  Start: {time2['start']}")
    print(f"  End: {time2['end']}")
    print(f"  Duration: {time2['duration_hours']:.1f} hours")
    assert time2["duration_hours"] == 0.5, "Format 2 duration failed"

    print("✓ All time window extraction tests passed")


def test_transform_and_validate():
    """Test report transformation and validation."""
    print("\n" + "="*80)
    print("TEST: Transform and Validate Report")
    print("="*80)

    # Load sample report
    with open("sample_report.json") as f:
        raw_report = json.load(f)

    print(f"Transforming report: {raw_report.get('report_id', 'N/A')}")

    # Transform
    transformed = transform_report(raw_report)

    print(f"✓ Transformation completed")
    print(f"  Location: {transformed.get('location')}")
    print(f"  Time start: {transformed.get('time_start')}")
    print(f"  Time end: {transformed.get('time_end')}")
    print(f"  Duration: {transformed.get('time_duration_hours', 0):.1f} hours")
    print(f"  Embedding dimensions: {len(transformed.get('text_embedding', []))}")

    # Validate
    validate_document(transformed)
    print("✓ Validation passed")

    # Check classification
    classification = transformed.get("incident_classification", {})
    print(f"\nIncident Classification:")
    print(f"  Severity: {classification.get('severity', 'N/A')}")
    print(f"  Urgency: {classification.get('urgency_score', 0):.2f}")
    types = classification.get("types", [])
    if types:
        print(f"  Types:")
        for t in types[:3]:
            if isinstance(t, dict):
                print(f"    - {t.get('type', 'N/A')} (confidence: {t.get('confidence', 0):.2f})")

    print("✓ All transformation and validation tests passed")


def test_similarity_scorer():
    """Test similarity scoring with mock data."""
    print("\n" + "="*80)
    print("TEST: Similarity Scoring")
    print("="*80)

    scorer = SimilarityScorer()

    # Test geo similarity
    geo_score = scorer.score_geo_similarity(
        query_lat=37.7749,
        query_lon=-122.4194,
        result_lat=37.7849,  # ~0.69 miles apart
        result_lon=-122.4094,
        max_distance_miles=5.0
    )
    print(f"\nGeo Similarity:")
    print(f"  Score: {geo_score['score']:.3f}")
    print(f"  {geo_score['explanation']}")
    assert 0.0 <= geo_score['score'] <= 1.0, "Geo score out of range"

    # Test time similarity
    time_score = scorer.score_time_similarity(
        query_start="2025-09-27T02:30:00Z",
        query_end="2025-09-27T03:00:00Z",
        result_start="2025-09-27T04:00:00Z",  # 1.75 hours apart
        result_end="2025-09-27T04:30:00Z",
        max_delta_hours=24.0
    )
    print(f"\nTime Similarity:")
    print(f"  Score: {time_score['score']:.3f}")
    print(f"  {time_score['explanation']}")
    assert 0.0 <= time_score['score'] <= 1.0, "Time score out of range"

    # Test text similarity
    text_score = scorer.score_text_similarity(es_score=15.5, max_es_score=20.0)
    print(f"\nText Similarity:")
    print(f"  Score: {text_score['score']:.3f}")
    print(f"  {text_score['explanation']}")
    assert 0.0 <= text_score['score'] <= 1.0, "Text score out of range"

    # Test type similarity
    type_score = scorer.score_type_similarity(
        query_type="assault",
        result_type="physical fight",
        result_classified_types=None
    )
    print(f"\nType Similarity:")
    print(f"  Score: {type_score['score']:.3f}")
    print(f"  {type_score['explanation']}")
    assert 0.0 <= type_score['score'] <= 1.0, "Type score out of range"

    print("\n✓ All similarity scoring tests passed")


def test_similarity_search_24hr():
    """Test similarity search with 24-hour window."""
    print("\n" + "="*80)
    print("TEST: Similarity Search (24-hour window)")
    print("="*80)

    # Load query report
    query_report = load_query_report("sample_report.json")
    print(f"Query report: {query_report.get('report_id')}")

    # Execute search
    results = execute_similarity_search(
        query_report,
        radius_miles=5.0,
        lookback_hours=24,
        limit=5
    )

    print(f"\nFound {len(results)} similar incidents")

    if results:
        print("\nTop 3 Results:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n  #{i} - Score: {result['final_score']:.3f}")
            print(f"      Report ID: {result['result'].get('report_id')}")
            print(f"      Type: {result['result'].get('incident_type')}")
            print(f"      Geo: {result['geo']['explanation']}")
            print(f"      Time: {result['time']['explanation']}")

    print("\n✓ 24-hour search test completed")


def test_similarity_search_1week():
    """Test similarity search with 1-week window."""
    print("\n" + "="*80)
    print("TEST: Similarity Search (1-week window)")
    print("="*80)

    # Load query report
    query_report = load_query_report("sample_report.json")

    # Execute search
    results = execute_similarity_search(
        query_report,
        radius_miles=10.0,
        lookback_days=7,
        limit=10
    )

    print(f"\nFound {len(results)} similar incidents")

    if results:
        print("\nScore distribution:")
        for i, result in enumerate(results[:5], 1):
            print(f"  #{i}: {result['final_score']:.3f}")

    print("\n✓ 1-week search test completed")


def test_custom_scorer_weights():
    """Test similarity search with custom scorer weights."""
    print("\n" + "="*80)
    print("TEST: Custom Scorer Weights")
    print("="*80)

    # Load query report
    query_report = load_query_report("sample_report.json")

    # Custom weights: prioritize location and time
    custom_weights = {
        "geo_weight": 0.35,
        "time_weight": 0.35,
        "text_weight": 0.2,
        "type_weight": 0.1
    }

    print(f"Using custom weights: {custom_weights}")

    results = execute_similarity_search(
        query_report,
        radius_miles=5.0,
        lookback_hours=24,
        limit=5,
        scorer_weights=custom_weights
    )

    print(f"\nFound {len(results)} similar incidents with custom weights")

    if results:
        print("\nTop result:")
        result = results[0]
        print(f"  Final Score: {result['final_score']:.3f}")
        print(f"  Geo Score: {result['geo']['score']:.3f}")
        print(f"  Time Score: {result['time']['score']:.3f}")
        print(f"  Text Score: {result['text']['score']:.3f}")
        print(f"  Type Score: {result['type']['score']:.3f}")

    print("\n✓ Custom scorer weights test completed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING SIMILARITY SEARCH TEST SUITE")
    print("="*80)

    try:
        # Unit tests
        test_extract_location()
        test_extract_time_window()
        test_transform_and_validate()
        test_similarity_scorer()

        # Integration tests (require Elasticsearch with data)
        print("\n" + "="*80)
        print("INTEGRATION TESTS (require Elasticsearch with indexed data)")
        print("="*80)

        test_similarity_search_24hr()
        test_similarity_search_1week()
        test_custom_scorer_weights()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    run_all_tests()
