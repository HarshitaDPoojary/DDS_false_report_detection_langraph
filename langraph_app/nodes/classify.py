"""
classify_node — incident type classification.

Strategy:
  1. Regex-based classifier (offline, deterministic) — always runs first.
  2. If regex returns nothing → LangChain chain with
     ChatOpenAI (gpt-4.1-mini) primary, ChatAnthropic fallback.

Returns: {incident_types, severity}
"""
from __future__ import annotations
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from legacy.incident_severity_score import get_incident_types, aggregate_severity
from langraph_app.config.constants import (
    CANONICAL_TYPES,
    CLASSIFICATION_SYSTEM_PROMPT,
    CLASSIFICATION_HUMAN_PROMPT,
)
from langraph_app.config.settings import get_settings


def _build_llm_chain():
    settings = get_settings()
    prompt = ChatPromptTemplate.from_messages([
        ("system", CLASSIFICATION_SYSTEM_PROMPT),
        ("human",  CLASSIFICATION_HUMAN_PROMPT),
    ])
    primary  = ChatOpenAI(model=settings.openai_model, temperature=0,
                          api_key=settings.openai_api_key or None)
    fallback = ChatAnthropic(model=settings.claude_model, temperature=0,
                             api_key=settings.get_anthropic_key() or None)
    return prompt | primary.with_fallbacks([fallback]) | JsonOutputParser()


_llm_chain = None


def _get_llm_chain():
    global _llm_chain
    if _llm_chain is None:
        _llm_chain = _build_llm_chain()
    return _llm_chain


def run(state: dict) -> dict:
    text = state.get("free_text", "")

    # Step 1: regex-based classification (fast, offline)
    types = get_incident_types(text)

    # Step 2: LLM fallback only if regex found nothing
    if not types or (len(types) == 1 and types[0].get("type") == "other"
                     and types[0].get("confidence", 0) <= 0.2):
        try:
            result = _get_llm_chain().invoke({
                "canonical_types": ", ".join(CANONICAL_TYPES),
                "report_text": text,
            })
            llm_types = result.get("types", []) if isinstance(result, dict) else []
            if llm_types:
                types = [
                    {
                        "type": t.get("type", "other") if t.get("type") in CANONICAL_TYPES else "other",
                        "confidence": round(float(t.get("confidence", 0.0)), 3),
                        "matches": [t.get("reason", "llm")],
                    }
                    for t in llm_types
                ]
        except Exception:
            pass  # keep regex result / "other" default

    severity_result = aggregate_severity(types)
    severity = severity_result.get("severity", "low") if isinstance(severity_result, dict) else "low"

    return {
        "incident_types": types,
        "severity": severity,
    }
