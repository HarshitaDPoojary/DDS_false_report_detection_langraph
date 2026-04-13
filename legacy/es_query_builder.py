"""
Elasticsearch Query Builder Module

Provides utility functions for building multi-dimensional similarity queries:
- Geo-spatial queries (geo_distance)
- Temporal queries (time overlap)
- Semantic text similarity (kNN search on embeddings)
- Incident type matching
"""

from datetime import datetime, timedelta


def build_geo_query(lat, lon, radius_miles=5.0):
    """
    Build geo_distance filter query.

    Args:
        lat: Latitude of query point
        lon: Longitude of query point
        radius_miles: Search radius in miles (default: 5.0)

    Returns:
        Dict with geo_distance filter query
    """
    return {
        "geo_distance": {
            "distance": f"{radius_miles}mi",
            "location": {
                "lat": lat,
                "lon": lon
            }
        }
    }


def build_time_query(start, end, lookback_hours=None, lookback_days=None):
    """
    Build time overlap query with lookback window.

    Args:
        start: Query time window start (ISO string or datetime)
        end: Query time window end (ISO string or datetime)
        lookback_hours: Optional lookback in hours (e.g., 24)
        lookback_days: Optional lookback in days (e.g., 7)

    Returns:
        Dict with bool query for time overlap
    """
    # Parse start/end if strings
    if isinstance(start, str):
        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
    if isinstance(end, str):
        end = datetime.fromisoformat(end.replace("Z", "+00:00"))

    # Apply lookback window
    if lookback_hours is not None:
        lookback_start = start - timedelta(hours=lookback_hours)
        lookback_end = end + timedelta(hours=lookback_hours)
    elif lookback_days is not None:
        lookback_start = start - timedelta(days=lookback_days)
        lookback_end = end + timedelta(days=lookback_days)
    else:
        # No lookback, use exact times
        lookback_start = start
        lookback_end = end

    # Time overlap query: result overlaps with query window
    # Overlap if: result_start <= query_end AND result_end >= query_start
    return {
        "bool": {
            "must": [
                {
                    "range": {
                        "time_start": {
                            "lte": lookback_end.isoformat()
                        }
                    }
                },
                {
                    "range": {
                        "time_end": {
                            "gte": lookback_start.isoformat()
                        }
                    }
                }
            ]
        }
    }


def build_text_similarity_query(query_embedding, boost=1.5, k=50, num_candidates=100):
    """
    Build kNN search query on text_embedding field for semantic similarity.

    Args:
        query_embedding: Pre-generated embedding vector (list of floats)
        boost: Relevance boost factor (default: 1.5)
        k: Number of results to retrieve (default: 50)
        num_candidates: Number of candidates for kNN search (default: 100)

    Returns:
        Dict with kNN query
    """
    return {
        "field": "text_embedding",
        "query_vector": query_embedding,
        "k": k,
        "num_candidates": num_candidates,
        "boost": boost
    }


def build_incident_type_query(incident_type, boost=2.0, include_related=True):
    """
    Build incident type matching query.

    Args:
        incident_type: Query incident type (string)
        boost: Relevance boost for type matches (default: 2.0)
        include_related: Whether to include related types (default: True)

    Returns:
        Dict with bool query for type matching
    """
    # Related incident types mapping
    related_types = {
        "assault": ["physical fight", "violence", "battery"],
        "theft": ["burglary", "robbery", "shoplifting", "stolen property"],
        "burglary": ["theft", "breaking and entering", "trespassing"],
        "vandalism": ["property damage", "graffiti", "destruction"],
        "harassment": ["stalking", "threatening behavior", "intimidation"],
        "fraud": ["scam", "identity theft", "financial crime"],
        "physical fight": ["assault", "violence", "altercation"]
    }

    # Build should clauses
    should_clauses = [
        {
            "term": {
                "incident_type": {
                    "value": incident_type,
                    "boost": boost
                }
            }
        }
    ]

    # Add related types if enabled
    if include_related and incident_type in related_types:
        for related in related_types[incident_type]:
            should_clauses.append({
                "term": {
                    "incident_type": {
                        "value": related,
                        "boost": boost * 0.5  # Lower boost for related types
                    }
                }
            })

    # Also check classified types (nested)
    should_clauses.append({
        "nested": {
            "path": "incident_classification.types",
            "query": {
                "match": {
                    "incident_classification.types.type": incident_type
                }
            },
            "score_mode": "max",
            "boost": boost * 0.8
        }
    })

    return {
        "bool": {
            "should": should_clauses,
            "minimum_should_match": 1
        }
    }


def build_combined_query(
    lat, lon, start, end, query_embedding, incident_type,
    radius_miles=5.0,
    lookback_hours=None,
    lookback_days=None,
    exclude_report_id=None,
    text_boost=1.5,
    type_boost=2.0,
    size=50
):
    """
    Build combined multi-dimensional similarity query.

    Args:
        lat: Query latitude
        lon: Query longitude
        start: Query time start
        end: Query time end
        query_embedding: Pre-generated text embedding vector
        incident_type: Query incident type
        radius_miles: Geo search radius (default: 5.0)
        lookback_hours: Time lookback in hours (optional)
        lookback_days: Time lookback in days (optional)
        exclude_report_id: Report ID to exclude (e.g., query report itself)
        text_boost: Boost for text similarity (default: 1.5)
        type_boost: Boost for type matching (default: 2.0)
        size: Max results to return (default: 50)

    Returns:
        Complete Elasticsearch query dict with knn and query sections
    """
    # Build geo filter (MUST match)
    geo_filter = build_geo_query(lat, lon, radius_miles)

    # Build time filter (MUST match)
    time_filter = build_time_query(start, end, lookback_hours, lookback_days)

    # Build type query (SHOULD match for boosting)
    type_query = build_incident_type_query(incident_type, type_boost)

    # Build kNN query for text similarity
    knn_query = build_text_similarity_query(
        query_embedding,
        boost=text_boost,
        k=size,
        num_candidates=size * 2
    )

    # Build must_not clause (exclude query report)
    must_not = []
    if exclude_report_id:
        must_not.append({
            "term": {
                "report_id": exclude_report_id
            }
        })

    # Combine into final query
    # Using kNN search with filters
    query = {
        "size": size,
        "knn": knn_query,
        "query": {
            "bool": {
                "filter": [
                    geo_filter,
                    time_filter
                ],
                "should": [
                    type_query
                ],
                "must_not": must_not
            }
        }
    }

    return query
