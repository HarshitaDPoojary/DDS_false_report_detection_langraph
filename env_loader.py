import os
from dotenv import load_dotenv

def load_es_config():
    load_dotenv()
    return {
        "host": os.getenv("ES_HOST"),
        "api_key": os.getenv("ES_API_KEY"),
        "index": os.getenv("ES_INDEX")
    }
