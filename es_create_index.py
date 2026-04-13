from elasticsearch import Elasticsearch
from env_loader import load_es_config

def create_index():
    config = load_es_config()
    es = Elasticsearch(config["host"], api_key=config["api_key"])
    
    mapping = {
        "mappings": {
            "properties": {
                "report_id": {"type": "keyword"},
                "incident_type": {"type": "keyword"},
                "free_text": {
                    "type": "text",
                    "term_vector": "yes",
                    "similarity": "BM25"
                },
                # Unified fields for querying
                "location": {"type": "geo_point"},  # NEW: Unified geo field [lon, lat]
                "time_start": {"type": "date"},     # NEW: Unified time start
                "time_end": {"type": "date"},       # NEW: Unified time end
                "time_midpoint": {"type": "date"},  # NEW: Calculated midpoint
                "time_duration_hours": {"type": "float"},  # NEW: Duration in hours

                # Semantic similarity
                "text_embedding": {                 # NEW: Semantic similarity vectors
                    "type": "dense_vector",
                    "dims": 384,                    # sentence-transformers/all-MiniLM-L12-v2
                    "index": True,
                    "similarity": "cosine"
                },

                # Pre-computed incident classification
                "incident_classification": {        # NEW: Pre-computed classifications
                    "type": "object",
                    "properties": {
                        "types": {"type": "nested"},
                        "severity": {"type": "keyword"},
                        "urgency_score": {"type": "float"}
                    }
                },

                # Original structures (preserved for compatibility)
                "who": {"type": "object"},
                "where": {
                    "type": "object",
                    "properties": {
                        "address": {"type": "text"},
                        "venue": {"type": "text"},
                        "coordinates": {
                            "properties": {
                                "lat": {"type": "float"},
                                "lon": {"type": "float"}
                            }
                        }
                    }
                },
                "when_window": {
                    "type": "object",
                    "properties": {
                        "earliest": {"type": "date"},
                        "latest": {"type": "date"}
                    }
                },
                "means": {"type": "text"},
                "first_second_hand": {"type": "keyword"},
                "attachments": {"type": "nested"},
                "targets": {"type": "keyword"},
                "created_at": {"type": "date"},
                "reporter": {"type": "object"}
            }
        }
    }

    es.indices.create(index=config["index"], body=mapping)
    print(f"Created index '{config['index']}'")

if __name__ == "__main__":
    create_index()
