from elasticsearch import Elasticsearch
from env_loader import load_es_config

config = load_es_config()
es = Elasticsearch(config["host"], api_key=config["api_key"])

# Get total count
count = es.count(index=config["index"])["count"]
print(f"Total documents in index: {count}")

# Get incident type breakdown
result = es.search(
    index=config["index"],
    body={
        "size": 0,
        "aggs": {
            "incident_types": {
                "terms": {
                    "field": "incident_type",
                    "size": 10
                }
            }
        }
    }
)

print("\nIncident type breakdown:")
for bucket in result["aggregations"]["incident_types"]["buckets"]:
    print(f"  {bucket['key']}: {bucket['doc_count']} reports")

# Get a sample document
sample = es.search(index=config["index"], body={"query": {"match_all": {}}, "size": 1})
print("\nSample document structure:")
import json
print(json.dumps(sample["hits"]["hits"][0]["_source"], indent=2))
