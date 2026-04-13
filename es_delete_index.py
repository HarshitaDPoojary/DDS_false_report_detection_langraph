from elasticsearch import Elasticsearch
from env_loader import load_es_config

def delete_index():
    config = load_es_config()
    es = Elasticsearch(config["host"], api_key=config["api_key"])

    if es.indices.exists(index=config["index"]):
        es.indices.delete(index=config["index"])
        print(f"Deleted index '{config['index']}'")
    else:
        print(f"Index '{config['index']}' does not exist")

if __name__ == "__main__":
    delete_index()
