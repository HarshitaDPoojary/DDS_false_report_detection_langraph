"""
Similarity Scoring Module

Multi-dimensional similarity scoring for incident reports:
- Geographic similarity (distance-based)
- Temporal similarity (time delta-based)
- Text similarity (from ES score)
- Incident type similarity
"""

from datetime import datetime
from geopy.distance import geodesic


class SimilarityScorer:
    """
    Multi-dimensional similarity scorer with configurable weights.

    All dimension scores are normalized to 0-1 range.
    Final score is weighted sum of dimension scores.
    """

    def __init__(
        self,
        geo_weight=0.2,
        time_weight=0.3,
        text_weight=0.3,
        type_weight=0.2
    ):
        """
        Initialize scorer with dimension weights.

        Args:
            geo_weight: Weight for geographic similarity (default: 0.2)
            time_weight: Weight for temporal similarity (default: 0.3)
            text_weight: Weight for text similarity (default: 0.3)
            type_weight: Weight for incident type similarity (default: 0.2)

        Note: Weights should sum to 1.0 for interpretability
        """
        self.geo_weight = geo_weight
        self.time_weight = time_weight
        self.text_weight = text_weight
        self.type_weight = type_weight

        # Validate weights sum to 1.0
        total = geo_weight + time_weight + text_weight + type_weight
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total}, not 1.0")

    def score_geo_similarity(
        self,
        query_lat, query_lon,
        result_lat, result_lon,
        max_distance_miles=5.0
    ):
        """
        Calculate geographic similarity score.

        Args:
            query_lat: Query latitude
            query_lon: Query longitude
            result_lat: Result latitude
            result_lon: Result longitude
            max_distance_miles: Maximum distance for normalization (default: 5.0)

        Returns:
            Dict with score (0-1), distance_miles, and explanation
        """
        # Calculate geodesic distance
        query_point = (query_lat, query_lon)
        result_point = (result_lat, result_lon)
        distance_miles = geodesic(query_point, result_point).miles

        # Normalize: closer = higher score
        # score = 1.0 at distance 0, score = 0.0 at max_distance or beyond
        score = max(0.0, 1.0 - (distance_miles / max_distance_miles))

        explanation = f"{distance_miles:.2f} miles away"

        return {
            "score": score,
            "distance_miles": distance_miles,
            "explanation": explanation
        }

    def score_time_similarity(
        self,
        query_start, query_end,
        result_start, result_end,
        max_delta_hours=24.0
    ):
        """
        Calculate temporal similarity score.

        Args:
            query_start: Query time window start (ISO string or datetime)
            query_end: Query time window end (ISO string or datetime)
            result_start: Result time window start (ISO string or datetime)
            result_end: Result time window end (ISO string or datetime)
            max_delta_hours: Maximum time delta for normalization (default: 24.0)

        Returns:
            Dict with score (0-1), time_delta_hours, and explanation
        """
        # Parse to datetime if strings
        if isinstance(query_start, str):
            query_start = datetime.fromisoformat(query_start.replace("Z", "+00:00"))
        if isinstance(query_end, str):
            query_end = datetime.fromisoformat(query_end.replace("Z", "+00:00"))
        if isinstance(result_start, str):
            result_start = datetime.fromisoformat(result_start.replace("Z", "+00:00"))
        if isinstance(result_end, str):
            result_end = datetime.fromisoformat(result_end.replace("Z", "+00:00"))

        # Calculate midpoints
        query_delta = (query_end - query_start) / 2
        query_midpoint = query_start + query_delta

        result_delta = (result_end - result_start) / 2
        result_midpoint = result_start + result_delta

        # Calculate time delta between midpoints
        time_delta = abs((result_midpoint - query_midpoint).total_seconds() / 3600)

        # Normalize: closer in time = higher score
        score = max(0.0, 1.0 - (time_delta / max_delta_hours))

        explanation = f"{time_delta:.1f} hours apart"

        return {
            "score": score,
            "time_delta_hours": time_delta,
            "explanation": explanation
        }

    def score_text_similarity(self, es_score, max_es_score=20.0):
        """
        Calculate text similarity score from Elasticsearch score.

        Args:
            es_score: Raw Elasticsearch _score from kNN or MLT query
            max_es_score: Maximum ES score for normalization (default: 20.0)

        Returns:
            Dict with score (0-1), raw_es_score, and explanation
        """
        # Normalize ES score to 0-1 range
        score = min(1.0, es_score / max_es_score)

        explanation = f"ES score: {es_score:.2f}"

        return {
            "score": score,
            "raw_es_score": es_score,
            "explanation": explanation
        }

    def score_type_similarity(
        self,
        query_type,
        result_type,
        result_classified_types=None
    ):
        """
        Calculate incident type similarity score.

        Args:
            query_type: Query incident type (string)
            result_type: Result incident type (string)
            result_classified_types: Optional list of classified types from classification

        Returns:
            Dict with score (0-1), match_type, and explanation
        """
        # Exact match
        if query_type == result_type:
            return {
                "score": 1.0,
                "match_type": "exact",
                "explanation": f"Exact match: {result_type}"
            }

        # Check classified types
        if result_classified_types:
            for classified in result_classified_types:
                if isinstance(classified, dict):
                    classified_type = classified.get("type", "")
                else:
                    classified_type = str(classified)

                if query_type.lower() in classified_type.lower():
                    return {
                        "score": 0.8,
                        "match_type": "classified",
                        "explanation": f"Classified as: {classified_type}"
                    }

        # Related types (basic mapping)
        related_types = {
            "assault": ["physical fight", "violence", "battery"],
            "theft": ["burglary", "robbery", "shoplifting"],
            "burglary": ["theft", "breaking and entering"],
            "vandalism": ["property damage", "graffiti"],
            "harassment": ["stalking", "threatening behavior"]
        }

        # Check if result_type is related to query_type
        if query_type in related_types:
            if result_type in related_types[query_type]:
                return {
                    "score": 0.5,
                    "match_type": "related",
                    "explanation": f"Related type: {result_type}"
                }

        # Check reverse (if query_type is related to result_type)
        if result_type in related_types:
            if query_type in related_types[result_type]:
                return {
                    "score": 0.5,
                    "match_type": "related",
                    "explanation": f"Related type: {result_type}"
                }

        # No match
        return {
            "score": 0.0,
            "match_type": "none",
            "explanation": f"Different type: {result_type}"
        }

    def compute_final_score(self, query_report, result_hit):
        """
        Compute final weighted similarity score for a result.

        Args:
            query_report: Query report dict with location, time, type, etc.
            result_hit: Elasticsearch hit dict with _source and _score

        Returns:
            Dict with final_score, dimension scores, and explanations
        """
        result_source = result_hit["_source"]
        es_score = result_hit.get("_score", 0.0)

        # Extract query parameters
        query_location = query_report.get("location", [])
        query_lat, query_lon = query_location[1], query_location[0]  # [lon, lat] -> lat, lon

        query_start = query_report.get("time_start")
        query_end = query_report.get("time_end")
        query_type = query_report.get("incident_type", "")

        # Extract result parameters
        result_location = result_source.get("location", [])
        result_lat, result_lon = result_location[1], result_location[0]

        result_start = result_source.get("time_start")
        result_end = result_source.get("time_end")
        result_type = result_source.get("incident_type", "")
        result_classified = result_source.get("incident_classification", {}).get("types", [])

        # Score each dimension
        geo_score_info = self.score_geo_similarity(query_lat, query_lon, result_lat, result_lon)
        time_score_info = self.score_time_similarity(query_start, query_end, result_start, result_end)
        text_score_info = self.score_text_similarity(es_score)
        type_score_info = self.score_type_similarity(query_type, result_type, result_classified)

        # Compute weighted final score
        final_score = (
            geo_score_info["score"] * self.geo_weight +
            time_score_info["score"] * self.time_weight +
            text_score_info["score"] * self.text_weight +
            type_score_info["score"] * self.type_weight
        )

        return {
            "final_score": final_score,
            "geo": geo_score_info,
            "time": time_score_info,
            "text": text_score_info,
            "type": type_score_info,
            "result": result_source
        }

    def rank_results(self, query_report, es_results):
        """
        Score and rank all Elasticsearch results.

        Args:
            query_report: Query report dict
            es_results: Elasticsearch search results (hits.hits)

        Returns:
            List of scored results, sorted by final_score descending
        """
        scored_results = []

        for hit in es_results:
            scored = self.compute_final_score(query_report, hit)
            scored_results.append(scored)

        # Sort by final_score descending
        scored_results.sort(key=lambda x: x["final_score"], reverse=True)

        return scored_results
