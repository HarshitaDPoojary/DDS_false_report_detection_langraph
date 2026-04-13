"""
Incident classification and severity scoring

Provides:
- keyword/regex based incident type classifier (deterministic, offline)
- optional LLM prompt builder and generic caller hook for semantic fallback
- severity mapping from types -> severity levels
- small demo showing Case 1 (local classifier) and Case 2 (LLM prompt + how to call)

This module is safe to import without network access. To enable LLM calls,
pass a callable `llm_client` (see `call_llm`) or provide OPENAI_API_KEY / HF_API_TOKEN.
"""

from __future__ import annotations

import os
import re
import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- load .env early (no-op if not installed) ---
try:
    from dotenv import load_dotenv, find_dotenv  # pip install python-dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass

# ----------------------------
# Defaults (modern, sensible)
# ----------------------------
DEFAULT_LLM_MODEL = "gpt-4.1-mini"  # fast/cost-effective JSON classification
PRO_LLM_MODEL = "gpt-4.1"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4.5-20250514"  # Claude 4.5 Sonnet (fall 2025)

# demo sample text used in multiple places
DEMO_REPORT_TEXT = "Multiple people reporting smoke and a loud explosion near the market."

# ---------------------------------------------------------------------------
# Canonical taxonomy: incident types mapped to regex patterns (lowercased)
# ---------------------------------------------------------------------------
INCIDENT_KEYWORDS: Dict[str, List[str]] = {
    "shooting": [r"\bshoot(?:ing|s|er)\b", r"\bshots?\b", r"\bgunfire\b", r"\bopen fire\b", r"\bactive shooter\b"],
    "bombing": [r"\bbomb\b", r"\bexplosion\b", r"\bexplosive\b", r"\bied\b", r"\bsuspicious device\b"],
    "stabbing": [r"\bstab(?:bing|bed|ber)\b", r"\bknife\b", r"\bcut\b"],
    "assault": [r"\bassault\b", r"\bbeat(?:ing)?\b", r"\battack(?:ed|s)?\b"],
    "robbery": [r"\brobber(?:y)?\b", r"\bmugging\b", r"\barmed robbery\b"],
    "burglary": [r"\bburglary\b", r"\bbreak-?in\b", r"\bhome invasion\b", r"\bbroke into\b", r"\bbroke in\b", r"\bforced (?:entry|open)\b", r"\bforced open\b"],
    "theft": [r"\btheft\b", r"\bstolen\b", r"\bshoplift(?:ing|ed)?\b", r"\bsteal(?:ing|s)?\b", r"\bstole\b", r"\bsnatch(?:ed)?\b", r"\btook\b", r"\brobbed\b"],
    "kidnapping": [r"\bkidnap(?:ping|ped)?\b", r"\babduct(?:ed|ion)?\b", r"\btaken\b"],
    "vandalism": [r"\bvandal(?:ism|ize|ised)\b", r"\bgraffiti\b", r"\bproperty damage\b"],
    "arson": [r"\barson\b", r"\bset (on )?fire\b", r"\bsuspicious fire\b"],
    "hit_and_run": [r"\bhit and run\b", r"\bhit-and-run\b", r"\bhit ?and ?run\b", r"\bfled the scene\b", r"\bfled\b"],
    "traffic_accident": [r"\baccident\b", r"\bcrash\b", r"\bcollision\b", r"\bpileup\b", r"\broad traffic incident\b", r"\bwas hit\b", r"\bhit a\b", r"\bhit an?\b", r"\bhit\b"],
    "sexual_assault": [r"\bsexual assault\b", r"\brape\b"],
    "suspicious_package": [r"\bsuspicious package\b", r"\bpackage\b left\b", r"\bunattended bag\b", r"\bbag with wires\b"],
    "protest": [r"\bprotest\b", r"\briot\b", r"\bdemonstration\b"],
    "hazmat": [r"\bchemical spill\b", r"\bgas leak\b", r"\bhazardous material\b", r"\bhazmat\b"],
    "hostage": [r"\bhostage\b", r"\bheld hostage\b"],
    "medical_emergency": [r"\bcollapsed\b", r"\bnot breathing\b", r"\bunconscious\b", r"\bcardiac arrest\b", r"\bmedical emergency\b"],
    "natural_disaster": [r"\bearthquake\b", r"\bflood\b", r"\btornado\b", r"\bstorm\b"],
    "cyber_incident": [r"\bhack(?:ed|ing)?\b", r"\bdata breach\b", r"\bphish(?:ing)?\b"],
}

# A canonical list used in LLM prompts and validation
CANONICAL_TYPES = list(INCIDENT_KEYWORDS.keys()) + ["other"]


def get_incident_types(
    text: str,
    top_k: int = 3,
    min_score: float = 0.1,
    enable_llm_fallback: bool = False,
    llm_client: Optional[Any] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> List[Dict[str, Any]]:
    """
    Classify an input report into one or more incident types.

    Returns a list of dicts: {"type": <canonical_type>, "confidence": float, "matches": [matched substrings]}

    - Uses rule-based regex matching and a simple scoring heuristic.
    - If no matches and enable_llm_fallback=True, will call `llm_client` if provided.
    """
    if not text or not text.strip():
        return []

    t = text.lower()
    scores: Dict[str, float] = defaultdict(float)
    matches: Dict[str, List[str]] = defaultdict(list)

    # Keyword matching
    for incident_type, patterns in INCIDENT_KEYWORDS.items():
        for pat in patterns:
            try:
                m = re.search(pat, t)
            except re.error:
                continue
            if m:
                matched_text = m.group(0).strip().lower()
                matches[incident_type].append(matched_text)
                # scoring heuristic
                if len(matched_text.split()) <= 2:
                    scores[incident_type] += 0.6
                else:
                    scores[incident_type] += 0.5
                # contextual bonuses
                if "armed" in matched_text or "active" in matched_text:
                    scores[incident_type] += 0.2
                if any(k in matched_text for k in ("fled", "flee", "ran away")):
                    scores[incident_type] += 0.3
                if any(k in matched_text for k in ("stole", "stolen", "snatch", "robbed", "took")):
                    scores[incident_type] += 0.15

    # Normalize and prepare output
    cap = 1.6
    normalized = []
    for itype, score in scores.items():
        seen = set()
        uniq_matches = []
        for m in matches.get(itype, []):
            if m not in seen:
                seen.add(m)
                uniq_matches.append(m)
        conf = min(score / cap, 1.0)
        normalized.append((itype, round(conf, 3), uniq_matches))

    # If no rule-based matches, optionally use LLM fallback
    if not normalized and enable_llm_fallback and llm_client is not None:
        prompt = build_llm_prompt_for_types(text)
        try:
            raw = call_llm(llm_client, prompt, llm_model)
            if isinstance(raw, str):
                parsed = json.loads(raw)
            elif isinstance(raw, dict):
                parsed = raw
            else:
                parsed = None

            if parsed and "types" in parsed and isinstance(parsed["types"], list):
                out = []
                for item in parsed["types"]:
                    ttype = item.get("type", "other")
                    if ttype not in CANONICAL_TYPES:
                        ttype = "other"
                    conf = float(item.get("confidence", 0.0))
                    reason = item.get("reason") or "llm"
                    out.append({"type": ttype, "confidence": round(conf, 3), "matches": [reason]})
                return out[:top_k]
        except Exception:
            pass

    # Sort and filter top_k
    normalized.sort(key=lambda x: x[1], reverse=True)
    results: List[Dict[str, Any]] = []
    for itype, conf, matched in normalized[:top_k]:
        if conf >= min_score:
            results.append({"type": itype, "confidence": conf, "matches": matched})

    if not results:
        results = [{"type": "other", "confidence": 0.2, "matches": []}]

    return results


def build_llm_prompt_for_types(text: str) -> str:
    canonical = ", ".join(CANONICAL_TYPES)
    instruction = (
        "You are an assistant that classifies short incident reports into one or more canonical incident types.\n"
        "Return ONLY a JSON object with a 'types' array. Each element must be an object with keys: 'type' (one of the canonical types),\n"
        "'confidence' (float 0.0-1.0), and 'reason' (short text explanation of what triggered the classification).\n\n"
        f"Canonical types: {canonical}.\n\n"
        "Rules:\n"
        "- Output must be parsable JSON and nothing else.\n"
        "- If multiple plausible types exist, include them with appropriate confidences.\n"
        "- If none apply, return [{\"type\":\"other\",\"confidence\":0.1,\"reason\":\"no clear indicators\"}].\n\n"
        "Examples:\n"
        "Input: 'Multiple shots fired outside the mall, people running, heard several pops and saw smoke.'\n"
        "Output: {\"types\":[{\"type\":\"shooting\",\"confidence\":0.93,\"reason\":\"mentions 'shots fired'\"}]}\n\n"
        "Input: 'There is a suspicious package at the subway entrance, unattended bag emitting an odor.'\n"
        "Output: {\"types\":[{\"type\":\"suspicious_package\",\"confidence\":0.88,\"reason\":\"'suspicious package' and 'unattended bag'\"}]}\n\n"
        "Now classify the following report and return only JSON:\n\n"
        f"Report:\n{text.strip()}\n"
    )
    return instruction


# -----------------------------------------
# OpenAI (>=1.x) client + calling helpers
# -----------------------------------------
def get_openai_client(api_key: Optional[str] = None):
    """
    Return an OpenAI client (>=1.x). Reads key from env if not provided.
    pip install openai>=1
    """
    from openai import OpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY")
    return OpenAI(api_key=key)


# def call_openai(prompt: str, model: str = DEFAULT_LLM_MODEL, api_key: Optional[str] = None) -> str:
#     """
#     Prefer the new Responses API; fall back to Chat Completions for compatibility.
#     Returns string content.
#     """
#     client = get_openai_client(api_key)

#     # Try Responses API
#     try:
#         resp = client.responses.create(
#             model=model,
#             input=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_output_tokens=600,
#         )
#         text = getattr(resp, "output_text", None)
#         if text:
#             return text
#         # stitch from parts if needed
#         if hasattr(resp, "output") and resp.output:
#             parts = []
#             for item in resp.output:
#                 if getattr(item, "content", None):
#                     for c in item.content:
#                         if getattr(c, "type", "") == "output_text":
#                             parts.append(getattr(c, "text", ""))
#             if parts:
#                 return "".join(parts)
#     except Exception:
#         pass

#     # Fallback: Chat Completions
#     try:
#         resp = client.chat.completions.create(
#             model=model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_tokens=600,
#         )
#         return resp.choices[0].message.content
#     except Exception as exc:
#         raise RuntimeError("OpenAI call failed (Responses + Chat Completions).") from exc


def call_openai(prompt: str, model: str = DEFAULT_LLM_MODEL, api_key: Optional[str] = None) -> str:
    """
    OpenAI >=1.x:
      - Try Responses API with plain-string input (preferred).
      - Fallback to Chat Completions with standard messages.
    Prints the underlying API error message if something goes wrong.
    """
    try:
        from openai import OpenAI
    except Exception as _exc:
        raise RuntimeError("OpenAI client not installed: pip install --upgrade openai") from _exc

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # 1) Responses API (string input)
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,              # <— plain string (not a chat dict)
            temperature=0.0,
            max_output_tokens=600,
        )
        # Most SDKs expose .output_text. If not, stitch content.
        text = getattr(resp, "output_text", None)
        if text:
            return text
        if hasattr(resp, "output") and resp.output:
            parts = []
            for item in resp.output:
                if getattr(item, "content", None):
                    for c in item.content:
                        if getattr(c, "type", "") == "output_text":
                            parts.append(getattr(c, "text", ""))
            if parts:
                return "".join(parts)
        # As a last resort
        return str(resp)
    except Exception as e_responses:
        # 2) Fallback: Chat Completions
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
            )
            return resp.choices[0].message.content
        except Exception as e_chat:
            # Show the real reason (model not found, quota exceeded, etc.)
            raise RuntimeError(
                f"OpenAI call failed. Responses error: {getattr(e_responses, 'message', str(e_responses))} | "
                f"ChatCompletions error: {getattr(e_chat, 'message', str(e_chat))}"
            ) from e_chat


def call_llm(llm_client: Callable[[str, str], Any] | Any, prompt: str, model: str) -> Any:
    """
    Accepts either:
      - a callable(prompt, model) -> str
      - an OpenAI client object with .responses or .chat.completions
    """
    # direct callable
    try:
        return llm_client(prompt, model)
    except TypeError:
        pass

    # OpenAI client object (>=1.x)
    try:
        if hasattr(llm_client, "responses"):
            resp = llm_client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_output_tokens=600,
            )
            txt = getattr(resp, "output_text", None)
            if txt:
                return txt
        if hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
            resp = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
            )
            return resp.choices[0].message.content
    except Exception as exc:
        raise RuntimeError("LLM call failed with provided client.") from exc

    raise RuntimeError("LLM call failed; unsupported client/callable shape.")


# -----------------------------------------
# Anthropic Claude helper
# -----------------------------------------
def get_claude_client(api_key: Optional[str] = None):
    """
    Return an Anthropic client.
    pip install anthropic
    """
    from anthropic import Anthropic
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    return Anthropic(api_key=key)


def call_claude(
    prompt: str,
    model: str = DEFAULT_CLAUDE_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = 600,
) -> str:
    """
    Call Anthropic Claude API.
    Returns the text content from the response.
    """
    try:
        from anthropic import Anthropic
    except Exception as _exc:
        raise RuntimeError("Anthropic client not installed: pip install anthropic") from _exc

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"))

    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from response
        if hasattr(response, "content") and response.content:
            # response.content is a list of ContentBlock objects
            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            return "".join(text_parts)
        return str(response)
    except Exception as exc:
        raise RuntimeError(f"Claude API call failed: {getattr(exc, 'message', str(exc))}") from exc


# -----------------------------------------
# Hugging Face Inference (router) helper
# -----------------------------------------
def call_huggingface_inference(
    prompt: str,
    hf_token: Optional[str] = None,
    model: str = "microsoft/Phi-3-mini-4k-instruct",
) -> str:
    """
    Use huggingface_hub.InferenceClient (works with HF router).
    pip install huggingface_hub
    
    Note: Free HuggingFace Inference API has limitations. Consider using:
    - Local transformers (call_local_transformers) for offline/free inference
    - Paid HF Inference Endpoints for production use
    - Small models that work with free tier: microsoft/Phi-3-mini-4k-instruct, google/flan-t5-base
    """
    from huggingface_hub import InferenceClient

    token = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError(
            "HF token missing (set HF_API_TOKEN or HUGGINGFACE_API_TOKEN). "
            "Get a free token at https://huggingface.co/settings/tokens"
        )

    client = InferenceClient(model=model, token=token)
    try:
        return client.text_generation(
            prompt,
            max_new_tokens=200,
            temperature=0.0,
            do_sample=False,
        )
    except Exception as exc:
        error_msg = str(exc).lower()
        # Provide specific guidance based on error type
        if "429" in error_msg or "rate limit" in error_msg:
            hint = "Rate limit exceeded. Try again later or use a paid endpoint."
        elif "403" in error_msg or "unauthorized" in error_msg:
            hint = "Token lacks permissions. Create a new token with 'Inference API' access at https://huggingface.co/settings/tokens"
        elif "503" in error_msg or "loading" in error_msg:
            hint = "Model is loading (cold start). Wait 20-30 seconds and retry, or use call_local_transformers() for instant local inference."
        elif "404" in error_msg:
            hint = f"Model '{model}' not found or not available via Inference API. Try: microsoft/Phi-3-mini-4k-instruct or google/flan-t5-base"
        else:
            hint = "Consider using call_local_transformers() for free offline inference instead."
        
        raise RuntimeError(
            f"HF Inference API failed: {exc}\n"
            f"Hint: {hint}"
        ) from exc


def llm_classify_types_with_fallbacks(
    text: str,
    provider_order: Optional[List[str]] = None,
    openai_model: Optional[str] = None,
    claude_model: Optional[str] = None,
    hf_model: Optional[str] = None,
    local_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    If regex found nothing, ask an LLM for canonical types.
    provider_order: list of "openai" | "claude" | "hf" | "local"
      default checks env LLM_PROVIDER_ORDER="openai,claude,hf,local"
    Returns same shape as get_incident_types(): [{"type", "confidence", "matches": ["reason"]}, ...]
    """
    provider_order = provider_order or (os.environ.get("LLM_PROVIDER_ORDER", "openai,claude,hf,local").split(","))

    prompt = build_llm_prompt_for_types(text)

    last_err = None
    for provider in [p.strip().lower() for p in provider_order if p.strip()]:
        try:
            if provider == "openai":
                model = openai_model or os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL)
                resp = call_openai(prompt, model=model)
                parsed = json.loads(resp) if isinstance(resp, str) else resp
            elif provider in ("claude", "anthropic"):
                model = claude_model or os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
                resp = call_claude(prompt, model=model)
                parsed = json.loads(resp) if isinstance(resp, str) else resp
            elif provider in ("hf", "huggingface"):
                model = hf_model or os.environ.get("HF_MODEL", "bigscience/bloomz-1b1")
                resp = call_huggingface_inference(prompt, model=model)
                parsed = json.loads(resp) if isinstance(resp, str) else resp
            elif provider == "local":
                model = local_model or os.environ.get("LOCAL_TRANSFORMERS_MODEL", "Qwen/Qwen2.5-3B-Instruct")
                resp = call_local_transformers(prompt, model_id=model)
                parsed = json.loads(resp) if isinstance(resp, str) else resp
            else:
                continue

            if parsed and isinstance(parsed, dict) and "types" in parsed:
                out = []
                for item in parsed["types"]:
                    ttype = item.get("type", "other")
                    if ttype not in CANONICAL_TYPES:
                        ttype = "other"
                    conf = float(item.get("confidence", 0.0))
                    reason = item.get("reason") or provider
                    out.append({"type": ttype, "confidence": round(conf, 3), "matches": [reason]})
                return out
        except Exception as exc:
            last_err = exc
            continue

    # If we got here, all providers failed
    if last_err:
        print(f"[LLM fallback] all providers failed: {last_err}")
    return [{"type": "other", "confidence": 0.2, "matches": []}]


def local_llm_simulator(prompt: str) -> str:
    """Tiny local simulator (no network)."""
    out = {"types": []}
    t = prompt.lower()
    if "explosion" in t or "bomb" in t:
        out["types"].append({"type": "bombing", "confidence": 0.95, "reason": "mentions explosion/bomb"})
    if "shots" in t or "shoot" in t or "gunfire" in t:
        out["types"].append({"type": "shooting", "confidence": 0.9, "reason": "mentions shots/gunfire"})
    if not out["types"]:
        out["types"].append({"type": "other", "confidence": 0.2, "reason": "no strong indicators (simulated)"})
    return json.dumps(out)

# =========================
# Local Transformers fallback (no external APIs)
# =========================
# pip install -U transformers accelerate
# For 4-bit on NVIDIA: pip install bitsandbytes
# Windows note: bitsandbytes has limited/experimental support on Windows.
# If 4-bit fails, we gracefully fall back to standard load or CPU.

from typing import Optional as _Optional
import re as _re
import json as _json
import os as _os

_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None

def _gpu_supports_bfloat16(_torch):
    try:
        return _torch.cuda.is_available() and _torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+ often bfloat16
    except Exception:
        return False

def _lazy_load_local_model(model_id: str = None, prefer_4bit: bool = True):
    """
    Lazy-load a local instruct model. If CUDA + bitsandbytes are available and prefer_4bit=True,
    load the model quantized to 4-bit (NF4) to dramatically reduce VRAM/RAM.
    """
    global _LOCAL_MODEL, _LOCAL_TOKENIZER
    if _LOCAL_MODEL is not None and _LOCAL_TOKENIZER is not None:
        return _LOCAL_MODEL, _LOCAL_TOKENIZER

    model_id = model_id or _os.environ.get("LOCAL_TRANSFORMERS_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    offload_dir = _os.environ.get("LOCAL_OFFLOAD_DIR","C:\hf_offload")  # <-- NEW

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if prefer_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig  # requires bitsandbytes
            compute_dtype = torch.bfloat16 if _gpu_supports_bfloat16(torch) else torch.float16
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_cfg,
                device_map="auto",
                low_cpu_mem_usage=True,
                attn_implementation="eager",
                offload_folder=offload_dir,      # <-- NEW (used when layers are offloaded)
            )
            return _LOCAL_MODEL, _LOCAL_TOKENIZER
        except Exception as _e4:
            print("[local 4-bit] Falling back from 4-bit quantization:", _e4)

    try:
        dtype = torch.bfloat16 if _gpu_supports_bfloat16(torch) else (torch.float16 if torch.cuda.is_available() else torch.float32)
        _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            offload_folder=offload_dir,          # <-- NEW (CPU/GPU offload folder)
        )
        if not torch.cuda.is_available():
            print("[local] Loaded on CPU; consider a smaller model (e.g., gemma-2-2b-it) for speed.")
        return _LOCAL_MODEL, _LOCAL_TOKENIZER
    except Exception as _e:
        raise RuntimeError(
            "Failed to load local Transformers model. "
            "Try a smaller model (LOCAL_TRANSFORMERS_MODEL=google/gemma-2-2b-it) or enable GPU."
        ) from _e

def _extract_first_json_block(text: str) -> _Optional[str]:
    """Best-effort: return the first JSON object/array found in the text, else None."""
    # object {...}
    obj = _re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=_re.DOTALL)
    if obj:
        return obj.group(0)
    # array [...]
    arr = _re.search(r"\[(?:[^\[\]]|(?R))*\]", text, flags=_re.DOTALL)
    if arr:
        return arr.group(0)
    return None


def call_local_transformers(
    prompt: str,
    model_id: _Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    do_sample: bool = False,
    force_json: bool = True,
    prefer_4bit: bool = True,   # NEW: try 4-bit first
) -> str:
    """
    Generate with a local model (Transformers). No internet/HF token required.
    If prefer_4bit=True and your environment supports bitsandbytes + CUDA,
    the model is loaded in 4-bit to save a lot of memory.

    Returns the raw text, or just the first JSON block if force_json=True and found.
    """
    model, tok = _lazy_load_local_model(model_id, prefer_4bit=prefer_4bit)

    import torch
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tok.eos_token_id,
        )
    out_text = tok.decode(out_ids[0], skip_special_tokens=True)

    # Many instruct models echo the prompt; strip if present
    if out_text.startswith(prompt):
        out_text = out_text[len(prompt):].lstrip()

    if force_json:
        maybe = _extract_first_json_block(out_text)
        if maybe:
            try:
                _ = _json.loads(maybe)  # validate
                return maybe
            except Exception:
                pass
    return out_text



# -----------------------------
# Severity mapping
# -----------------------------
SEVERITY_ORDER = ["low", "medium", "high", "critical"]

TYPE_TO_SEVERITY: Dict[str, str] = {
    "shooting": "critical",
    "bombing": "critical",
    "stabbing": "high",
    "assault": "high",
    "robbery": "high",
    "burglary": "medium",
    "theft": "low",
    "kidnapping": "critical",
    "vandalism": "low",
    "arson": "high",
    "hit_and_run": "high",
    "traffic_accident": "medium",
    "sexual_assault": "critical",
    "suspicious_package": "high",
    "protest": "medium",
    "hazmat": "critical",
    "hostage": "critical",
    "medical_emergency": "high",
    "natural_disaster": "critical",
    "cyber_incident": "medium",
    "other": "low",
}


def map_type_to_severity(itype: str) -> str:
    return TYPE_TO_SEVERITY.get(itype, "low")


def aggregate_severity(classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given classifications like [{type, confidence, matches}], return:
    {"severity": <level>, "reason": <top_reason>, "details": classifications}
    """
    if not classifications:
        return {"severity": "unknown", "reason": "no data", "details": []}

    best_weight = -1.0
    best_sev = "unknown"
    best_item = None
    for item in classifications:
        itype = item.get("type", "other")
        conf = float(item.get("confidence", 0.0))
        sev = map_type_to_severity(itype)
        idx = SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else 0
        weighted = idx * conf
        if best_item is None or weighted > best_weight:
            best_weight = weighted
            best_item = item
            best_sev = sev

    if best_item:
        return {"severity": best_sev, "reason": best_item.get("matches") or best_item.get("type"), "details": classifications}

    return {"severity": "unknown", "reason": "no clear mapping", "details": classifications}


# -----------------------------
# Urgency / weighted scoring
# -----------------------------
BASE_SCORE_BY_TYPE: Dict[str, float] = {
    "shooting": 9.0,
    "bombing": 10.0,
    "stabbing": 6.5,
    "assault": 6.0,
    "robbery": 5.5,
    "burglary": 4.0,
    "theft": 2.0,
    "kidnapping": 9.5,
    "vandalism": 2.0,
    "arson": 8.0,
    "hit_and_run": 6.0,
    "traffic_accident": 5.0,
    "sexual_assault": 9.0,
    "suspicious_package": 8.5,
    "protest": 4.0,
    "hazmat": 9.5,
    "hostage": 10.0,
    "medical_emergency": 7.0,
    "natural_disaster": 10.0,
    "cyber_incident": 3.0,
    "other": 1.0,
}


def _extract_location_weight(text: str) -> Tuple[float, str]:
    t = text.lower()
    mapping = [
        (r"\bbank\b", ("bank", 4.0)),
        (r"\b(s?upermarket|mall|retail|store|shop)\b", ("retail", 2.5)),
        (r"\b(home|house|apartment|residence)\b", ("home", 2.0)),
        (r"\b(warehouse|office|closed office)\b", ("warehouse", 1.0)),
        (r"\b(school|hospital)\b", ("sensitive", 3.0)),
        (r"\b(subway|bus|train|transit|station)\b", ("transit", 3.0)),
    ]
    for pat, (label, weight) in mapping:
        if re.search(pat, t):
            return weight, label
    return 0.0, "unknown"


def _extract_weapon_weight(text: str) -> Tuple[float, str]:
    t = text.lower()
    if re.search(r"\b(gun|firearm|rifle|pistol|shotgun|shooting)\b", t):
        return 3.0, "firearm"
    if re.search(r"\b(knife|blade|stabbed|stabbing|bat|club|blunt)\b", t):
        return 1.5, "knife/blunt"
    return 0.0, "none"


def _extract_suspect_count_weight(text: str) -> Tuple[float, int]:
    t = text.lower()
    m = re.search(r"(\b(\d+)\b) ?(suspects|people|persons|men|women|individuals)?", t)
    if m:
        try:
            n = int(m.group(2))
        except Exception:
            n = 1
        if n == 1:
            return 0.5, n
        if 2 <= n <= 3:
            return 1.0, n
        if n >= 4:
            return 2.0, n
    if re.search(r"\bone\b( suspect)?\b", t):
        return 0.5, 1
    if re.search(r"\btwo|three|2|3\b", t):
        return 1.0, 2
    if re.search(r"\bfour|five|several|many\b", t):
        return 2.0, 4
    return 0.0, 0


def _extract_time_weight(text: str) -> Tuple[float, str]:
    t = text.lower()
    if re.search(r"\b(night|tonight|after dark|after-dark|9pm|10pm|11pm|12am|1am|2am|3am|4am|5am)\b", t):
        return 1.5, "night"
    return 0.0, "day"


def _extract_vulnerability_weight(text: str) -> Tuple[float, str]:
    t = text.lower()
    if re.search(r"\b(children|child|kids|elderly|old people|senior|baby|infant)\b", t):
        return 2.0, "children/elderly"
    if re.search(r"\badults? (inside|home|present)|people inside|occupants|people home\b", t):
        return 1.0, "adults"
    if re.search(r"\b(unoccupied|vacant|empty|no one inside)\b", t):
        return 0.0, "unoccupied"
    return 0.0, "unknown"


def calculate_urgency(classifications: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
    if not classifications:
        return {"urgency": 0.0, "breakdown": [], "reason": "no classifications"}

    top = max(classifications, key=lambda x: float(x.get("confidence", 0.0)))
    ttype = top.get("type", "other")
    base = BASE_SCORE_BY_TYPE.get(ttype, 1.0)

    if ttype in ("bombing", "hostage", "natural_disaster"):
        return {"urgency": 10.0, "reason": ttype, "breakdown": [{"base": base, "override": "max for type"}]}

    loc_w, loc_label = _extract_location_weight(text)
    weap_w, weap_label = _extract_weapon_weight(text)
    sus_w, sus_n = _extract_suspect_count_weight(text)
    time_w, time_label = _extract_time_weight(text)
    vuln_w, vuln_label = _extract_vulnerability_weight(text)

    total = min(base + loc_w + weap_w + sus_w + time_w + vuln_w, 10.0)

    breakdown = [
        {"component": "base", "type": ttype, "value": base},
        {"component": "location", "label": loc_label, "value": loc_w},
        {"component": "weapon", "label": weap_label, "value": weap_w},
        {"component": "suspects", "count": sus_n, "value": sus_w},
        {"component": "time", "label": time_label, "value": time_w},
        {"component": "vulnerability", "label": vuln_label, "value": vuln_w},
    ]

    return {"urgency": round(total, 2), "reason": ttype, "breakdown": breakdown}


# ---------------------------------------------------------------------------
# Demo / example usage
# ---------------------------------------------------------------------------
def _demo_local():
    samples = [
        "I just heard multiple shots fired near the corner of 5th and Main. People are screaming.",
        "There is a suspicious package at the subway entrance, an unattended bag with wires.",
        "Someone broke into the house and stole jewelry. The back door was forced open.",
        "Car hit a bicyclist and fled the scene.",
    ]

    print("--- Case 1: Local deterministic classifier ---")
    for s in samples:
        types = get_incident_types(s)
        severity = aggregate_severity(types)
        urgency = calculate_urgency(types, s)
        print("Report:", s)
        print("Classification:", types)
        print("Severity:", severity)
        print("Urgency:", urgency)
        print()


def _demo_llm_prompt_only():
    text = DEMO_REPORT_TEXT
    prompt = build_llm_prompt_for_types(text)
    print("--- Case 2: LLM prompt (no network call) ---")
    print("Report:", text)
    print("Prompt to send to LLM (JSON-only expected):\n")
    print(prompt)
    print('\nExpected output example:')
    print(json.dumps({"types": [{"type": "bombing", "confidence": 0.92, "reason": "mentions explosion and smoke"}]}, indent=2))


# if __name__ == "__main__":
#     # Case 1: Local deterministic classifier
#     _demo_local()

#     # Optional: Live LLM call (OpenAI)
#     # print("\n--- Optional: Live LLM call demo (OpenAI) ---")
#     openai_key = os.environ.get("OPENAI_API_KEY")
#     # print(openai_key and "OpenAI API key detected; attempting live call." or "No OpenAI API key detected; skipping live call.")
#     if openai_key:
#         try:
#             client = get_openai_client(openai_key)
#             prompt = build_llm_prompt_for_types(DEMO_REPORT_TEXT)
#             print("\nAttempting live LLM call using OpenAI...")
#             resp = call_llm(client, prompt, model=DEFAULT_LLM_MODEL)
#             print("LLM raw response:", resp)
#         except Exception as exc:
#             print("OpenAI live call failed or not configured:", exc)

#     # Case 2: LLM wrappers demo (local + optional providers)
#     print("\n--- Case 2: LLM wrappers demo ---")

#     # Local simulator
#     print("\nLocal LLM simulator output:")
#     try:
#         print("TEXT INPUT:", DEMO_REPORT_TEXT)
#         print(local_llm_simulator(build_llm_prompt_for_types(DEMO_REPORT_TEXT)))
#     except Exception as exc:
#         print("Local simulator failed:", exc)

#     # OpenAI wrapper direct call (if token present)
#     if os.environ.get("OPENAI_API_KEY"):
#         print("\nOpenAI wrapper output:")
#         try:
#             print("TEXT INPUT:", DEMO_REPORT_TEXT)
#             print(call_openai(build_llm_prompt_for_types(DEMO_REPORT_TEXT), model=DEFAULT_LLM_MODEL))
#         except Exception as exc:
#             print("OpenAI wrapper failed:", exc)

#     # HuggingFace inference wrapper (requires token with Inference API permission)
#     if os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN"):
#         print("\nHuggingFace inference output:")
#         try:
#             print(call_huggingface_inference(build_llm_prompt_for_types(DEMO_REPORT_TEXT)))
#         except Exception as exc:
#             print("HuggingFace inference failed:", exc)

#     # Urgency demo
#     print("\nDemo text urgency (from local classifier):")
#     cls = get_incident_types(DEMO_REPORT_TEXT)
#     print(calculate_urgency(cls, DEMO_REPORT_TEXT))


if __name__ == "__main__":
    # 1) Deterministic classifier demo
    _demo_local()

    # 2) Build once and reuse the same prompt everywhere
    text_input = DEMO_REPORT_TEXT
    prompt = build_llm_prompt_for_types(text_input)

    print("\n--- Case 2: LLM wrappers demo (show each provider independently) ---")

    # 2a) Local LLM simulator (always available; no deps)
    print("\nLocal LLM simulator output:")
    try:
        print("TEXT INPUT:", text_input)
        print(local_llm_simulator(prompt))
    except Exception as exc:
        print("Local simulator failed:", exc)

    # 2b) OpenAI (if key present)
    print("\nOpenAI wrapper output:")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            print("TEXT INPUT:", text_input)
            print(call_openai(prompt, model=DEFAULT_LLM_MODEL))
        except Exception as exc:
            print("OpenAI wrapper failed:", exc)
    else:
        print("Skipped: no OPENAI_API_KEY in environment.")

    # 2c) Claude (Anthropic) (if key present)
    print("\nClaude (Anthropic) wrapper output:")
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"):
        try:
            print("TEXT INPUT:", text_input)
            # You can override model with env CLAUDE_MODEL; default is claude-4.5-sonnet (fall 2025)
            claude_model = os.environ.get("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL)
            print(call_claude(prompt, model=claude_model))
        except Exception as exc:
            print("Claude wrapper failed:", exc)
    else:
        print("Skipped: no ANTHROPIC_API_KEY or CLAUDE_API_KEY in environment.")

    # 2d) Hugging Face Inference (if token present)
    print("\nHuggingFace inference output:")
    hf_success = False
    if os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN"):
        try:
            # You can override with env HF_MODEL; otherwise uses microsoft/Phi-3-mini-4k-instruct
            hf_model = os.environ.get("HF_MODEL", "microsoft/Phi-3-mini-4k-instruct")
            result = call_huggingface_inference(prompt, model=hf_model)
            print(result)
            hf_success = True
        except Exception as exc:
            print(f"HuggingFace API failed: {exc}")
            print("\n→ Falling back to Local Transformers for free offline inference...")
    else:
        print("Skipped: no HF_API_TOKEN / HUGGINGFACE_API_TOKEN in environment.")
        print("→ Will use Local Transformers instead (free, offline).")
    
    # Auto-fallback to local transformers if HF failed or was skipped
    if not hf_success:
        try:
            print("\nAttempting local inference with small model (google/gemma-2-2b-it)...")
            local_result = call_local_transformers(
                prompt,
                model_id="google/gemma-2-2b-it",
                max_new_tokens=256,
                temperature=0.0,
                force_json=True,
            )
            print("Local Transformers output:", local_result)
        except Exception as local_exc:
            print(f"Local fallback also failed: {local_exc}")
            print("Note: First run downloads the model (~5GB). Ensure you have transformers installed:")
            print("  pip install transformers accelerate")

    # 2e) Local Transformers (fully offline if model is cached)
    print("\nLocal Transformers output:")
    if "call_local_transformers" in globals():
        try:
            # OPTIONAL: choose a local model via env LOCAL_TRANSFORMERS_MODEL
            # e.g., set LOCAL_TRANSFORMERS_MODEL=google/gemma-2-2b-it for CPU-friendly
            result = call_local_transformers(
                prompt,
                model_id=os.environ.get("LOCAL_TRANSFORMERS_MODEL", "google/gemma-2-2b-it"),
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                force_json=True,   # return only the first valid JSON block if present
            )
            print(result)
        except Exception as exc:
            print("Local Transformers failed (ensure transformers/accelerate installed and model fits your HW):", exc)
    else:
        print("Skipped: call_local_transformers not defined (paste the local Transformers helper section first).")

    # 3) Urgency demo (still uses the regex classifier only)
    print("\nDemo text urgency (from local classifier):")
    cls = get_incident_types(text_input)
    print(calculate_urgency(cls, text_input))
