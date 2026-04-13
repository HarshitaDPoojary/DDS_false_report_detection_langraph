"""
embed_node — generates a 384-dimensional text embedding using
HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L12-v2).
Singleton model loaded once at import time.
"""
from __future__ import annotations
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def _get_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )


def run(state: dict) -> dict:
    text = state.get("free_text", "")
    if not text.strip():
        return {"text_embedding": [0.0] * 384}
    embedding = _get_model().embed_query(text)
    return {"text_embedding": embedding}
