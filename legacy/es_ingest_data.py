import json
import argparse
import os
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers
from env_loader import load_es_config
from sentence_transformers import SentenceTransformer
from incident_severity_score import get_incident_types, aggregate_severity, calculate_urgency

# Model configuration (set via environment variable or default)
# Options from smallest to largest:
# - "paraphrase-MiniLM-L3-v2"          # 17MB, 17M params, 384 dims (smallest, fast)
# - "all-MiniLM-L12-v2"                # 120MB, 33M params, 384 dims (default, balanced)
# - "all-mpnet-base-v2"                # 420MB, 109M params, 768 dims (best quality, needs more RAM)
# - "sentence-transformers/all-MiniLM-L6-v2" # 90MB, 22M params, 384 dims (good middle ground)

DEFAULT_MODEL = "all-MiniLM-L12-v2"  # Default: balanced size/quality
# For better infrastructure:
# DEFAULT_MODEL = "all-mpnet-base-v2"  # Uncomment for production (420MB, 768 dims)

_embedding_model = None

def get_embedding_model():
    """Lazy-load embedding model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
        print(f"Loading embedding model: {model_name}...")
        _embedding_model = SentenceTransformer(model_name)
        print(f"Model loaded successfully!")
    return _embedding_model

def generate_text_embedding(free_text):
    """Generate embedding for text."""
    model = get_embedding_model()
    embedding = model.encode(free_text).tolist()  # [0.123, -0.456, ...]
    return embedding

def extract_location(report):
    """
    Normalize coordinates to [lon, lat] format for Elasticsearch geo_point.
    Handles 3 formats:
    - Format 1: where.coordinates.{lat, lon} (reports_cluster.json)
    - Format 2: where.geo array [lat, lon] (sample_report.json)
    - Format 3: where.geo_point GeoJSON (sample_report.json)
    """
    where = report.get("where", {})

    # Format 1: coordinates.{lat, lon}
    if "coordinates" in where and isinstance(where["coordinates"], dict):
        lat = where["coordinates"].get("lat")
        lon = where["coordinates"].get("lon")
        if lat is not None and lon is not None:
            return [lon, lat]  # ES geo_point expects [lon, lat]

    # Format 2: geo array [lat, lon]
    if "geo" in where and isinstance(where["geo"], list) and len(where["geo"]) == 2:
        lat, lon = where["geo"]
        return [lon, lat]

    # Format 3: geo_point GeoJSON
    if "geo_point" in where and isinstance(where["geo_point"], dict):
        coords = where["geo_point"].get("coordinates", [])
        if len(coords) == 2:
            return coords  # Already [lon, lat]

    raise ValueError(f"Could not extract location from report: {report.get('report_id', 'unknown')}")

def extract_time_window(report):
    """
    Normalize time window to start/end/midpoint/duration.
    Handles 2 formats:
    - Format 1: when_window.earliest / latest (reports_cluster.json)
    - Format 2: when_window.start_iso / end_iso (sample_report.json)
    Returns dict with start, end, midpoint (ISO strings), duration_hours (float)
    """
    when = report.get("when_window", {})

    # Format 1: earliest/latest
    if "earliest" in when and "latest" in when:
        start = when["earliest"]
        end = when["latest"]
    # Format 2: start_iso/end_iso
    elif "start_iso" in when and "end_iso" in when:
        start = when["start_iso"]
        end = when["end_iso"]
    else:
        raise ValueError(f"Could not extract time window from report: {report.get('report_id', 'unknown')}")

    # Parse to datetime for calculations
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))

    # Calculate midpoint
    delta = (end_dt - start_dt) / 2
    midpoint_dt = start_dt + delta

    # Calculate duration in hours
    duration_hours = (end_dt - start_dt).total_seconds() / 3600

    return {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "midpoint": midpoint_dt.isoformat(),
        "duration_hours": duration_hours
    }

def classify_incident(free_text):
    """
    Use incident_severity_score.py for classification.
    Returns structured classification object.
    """
    # Get incident types with confidence
    types = get_incident_types(free_text)

    # Aggregate severity - returns dict with 'severity' key
    severity_result = aggregate_severity(types)
    severity_value = severity_result.get("severity", "unknown") if isinstance(severity_result, dict) else str(severity_result)

    # Calculate urgency - returns dict with 'urgency' key
    urgency_result = calculate_urgency(types, free_text)
    urgency_value = urgency_result.get("urgency", 0.0) if isinstance(urgency_result, dict) else float(urgency_result)

    return {
        "types": types,  # List of dicts with type, confidence, matches
        "severity": severity_value,  # String: low, medium, high, critical
        "urgency_score": urgency_value  # Float: 0-10
    }

def transform_report(raw_report):
    """
    Transform raw report to include unified fields and embeddings.
    Returns transformed document ready for indexing.
    """
    # Extract and normalize location
    location = extract_location(raw_report)

    # Extract and normalize time window
    time_info = extract_time_window(raw_report)

    # Classify incident
    free_text = raw_report.get("free_text", "")
    classification = classify_incident(free_text)

    # Generate text embedding
    text_embedding = generate_text_embedding(free_text)

    # Build transformed document
    transformed = {
        **raw_report,  # Preserve original fields
        "location": location,
        "time_start": time_info["start"],
        "time_end": time_info["end"],
        "time_midpoint": time_info["midpoint"],
        "time_duration_hours": time_info["duration_hours"],
        "text_embedding": text_embedding,
        "incident_classification": classification
    }

    return transformed

def validate_document(doc):
    """
    Validate that required fields exist and are correctly formatted.
    Raises ValueError if validation fails.
    """
    required_fields = ["report_id", "location", "time_start", "time_end", "free_text"]

    for field in required_fields:
        if field not in doc:
            raise ValueError(f"Missing required field: {field}")

    # Validate location is [lon, lat] array
    if not isinstance(doc["location"], list) or len(doc["location"]) != 2:
        raise ValueError(f"Invalid location format: {doc['location']}")

    # Validate time fields are strings (ISO format)
    if not isinstance(doc["time_start"], str) or not isinstance(doc["time_end"], str):
        raise ValueError(f"Invalid time format")

    # Validate text_embedding is a list
    if not isinstance(doc.get("text_embedding"), list):
        raise ValueError(f"Invalid text_embedding format")

    return True

def ingest_reports(file_path):
    config = load_es_config()
    es = Elasticsearch(config["host"], api_key=config["api_key"])

    with open(file_path) as f:
        data = json.load(f)

    # Handle both single report and array of reports
    if isinstance(data, dict):
        reports = [data]
    elif isinstance(data, list):
        reports = data
    else:
        raise ValueError("Invalid JSON format: expected dict or list")

    print(f"Processing {len(reports)} reports...")

    # Transform and validate reports
    actions = []
    success_count = 0
    error_count = 0

    for i, raw_report in enumerate(reports):
        try:
            # Transform report
            transformed = transform_report(raw_report)

            # Validate document
            validate_document(transformed)

            # Add to bulk actions
            actions.append({
                "_index": config["index"],
                "_id": transformed["report_id"],
                "_source": transformed
            })

            success_count += 1
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(reports)} reports...")

        except Exception as e:
            error_count += 1
            print(f"  ERROR processing report {raw_report.get('report_id', 'unknown')}: {e}")

    # Execute bulk indexing
    if actions:
        print(f"\nIndexing {len(actions)} documents...")
        try:
            success, errors = helpers.bulk(es, actions, raise_on_error=False, raise_on_exception=False)
            print(f"Successfully indexed {success} documents")

            if errors:
                print(f"\nBulk indexing errors:")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"  {error}")

        except Exception as e:
            print(f"Bulk indexing failed: {e}")

    print(f"\nFinal stats:")
    print(f"  Successfully processed: {success_count}")
    if error_count > 0:
        print(f"  Failed to process: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to JSON file containing reports")
    args = parser.parse_args()
    ingest_reports(args.file)
