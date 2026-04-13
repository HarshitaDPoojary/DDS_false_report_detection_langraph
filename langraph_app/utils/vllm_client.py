"""
vllm_client — shared vision LLM client with local vLLM → NVIDIA NIM fallback.

Priority:
  1. Local vLLM server (VLLM_BASE_URL) — checked via /health endpoint at call time
  2. NVIDIA NIM (NIM_BASE_URL + NIM_API_KEY) — used when local vLLM is unreachable
  3. RuntimeError if neither is configured

Both backends expose an OpenAI-compatible /v1/chat/completions endpoint, so
langchain_openai.ChatOpenAI works with both via the base_url parameter.

GPU advisory check (check_gpu_vram) is a standalone helper for startup scripts —
it is NOT called during inference. The /health probe is the runtime gate.
"""
from __future__ import annotations

import subprocess
import urllib.error
import urllib.request

from langchain_openai import ChatOpenAI

from langraph_app.config.settings import get_settings


def get_vllm_client(max_tokens: int = 1024) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance pointing at the best available vision backend.

    Tries local vLLM first (3-second health probe). Falls back to NVIDIA NIM
    if local is unreachable and NIM_API_KEY is set.

    Raises RuntimeError if neither backend is available.
    """
    s = get_settings()

    # ── 1. Try local vLLM ────────────────────────────────────────────────────
    if s.vllm_base_url and s.vllm_api_key:
        if _is_vllm_healthy(s.vllm_base_url):
            return ChatOpenAI(
                base_url=s.vllm_base_url,
                api_key=s.vllm_api_key,
                model=s.vllm_vision_model,
                max_tokens=max_tokens,
            )

    # ── 2. Fallback to NVIDIA NIM ─────────────────────────────────────────────
    if s.nim_api_key and s.nim_base_url:
        return ChatOpenAI(
            base_url=s.nim_base_url,
            api_key=s.nim_api_key,
            model=s.nim_vision_model,
            max_tokens=max_tokens,
        )

    raise RuntimeError(
        "No vision backend available. "
        "Set VLLM_BASE_URL + VLLM_API_KEY for local vLLM, "
        "or NIM_BASE_URL + NIM_API_KEY for NVIDIA NIM."
    )


def _is_vllm_healthy(base_url: str) -> bool:
    """Probe /health on the vLLM server. Returns True if reachable and healthy."""
    # Strip trailing /v1 to reach the server root
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    health_url = f"{root}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def check_gpu_vram() -> dict:
    """
    Advisory GPU VRAM check via nvidia-smi.

    Returns a dict with:
      total_mb  — total VRAM across all GPUs (MB)
      free_mb   — free VRAM across all GPUs (MB)
      sufficient_for_qwen2vl_7b — True if free_mb >= 14,000 (~14GB needed)

    Returns zeros if nvidia-smi is unavailable (CPU-only machine).
    For startup scripts only — not called during inference.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        total_mb = 0
        free_mb = 0
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                total_mb += int(parts[0].strip())
                free_mb += int(parts[1].strip())
        return {
            "total_mb": total_mb,
            "free_mb": free_mb,
            "sufficient_for_qwen2vl_7b": free_mb >= 14_000,
        }
    except Exception:
        return {"total_mb": 0, "free_mb": 0, "sufficient_for_qwen2vl_7b": False}
