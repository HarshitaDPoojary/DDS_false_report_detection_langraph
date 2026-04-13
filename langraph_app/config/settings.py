"""
Application settings loaded from .env file.
Replaces env_loader.py with pydantic-settings BaseSettings.
"""
from __future__ import annotations
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Elasticsearch — required
    es_host: str
    es_api_key: str
    es_index: str = "incident_reports"

    # MongoDB — required
    mongo_uri: str
    mongo_db: str = "incident_reports"

    # OpenAI — required
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"
    openai_vision_model: str = "gpt-4o"

    # Anthropic (fallback for classification) — optional
    anthropic_api_key: str = ""
    claude_api_key: str = ""          # alias — whichever is set
    claude_model: str = "claude-sonnet-4-5-20250514"

    # Embeddings
    embedding_model: str = "all-MiniLM-L12-v2"

    # vLLM (self-hosted OpenAI-compatible inference server) — optional; NIM used as fallback
    vllm_base_url: str = ""
    vllm_vision_model: str = ""
    vllm_api_key: str = ""

    # NVIDIA NIM fallback (used when local vLLM is unreachable) — optional
    nim_base_url: str = ""
    nim_api_key: str = ""
    nim_vision_model: str = ""

    # Face recognition (InsightFace against known-offender DB) — optional
    face_recognition_enabled: bool = False
    known_offender_db_path: str = ""
    face_match_threshold: float = 0.60

    # Subject-of-Concern hashing salt — required
    soc_salt: str

    # API authentication key for analyst-facing endpoints — required
    api_key: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def get_anthropic_key(self) -> str:
        """Return whichever Anthropic key is set."""
        return self.anthropic_api_key or self.claude_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
