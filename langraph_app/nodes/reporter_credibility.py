"""
reporter_credibility_node — score reporter reliability using SOC history and
metadata collected at submission time.

Inputs (from raw_report.reporter, populated by the API layer from HTTP headers):
  anonymous           bool
  acct_age_days       int   — 0 for anonymous/new accounts
  prior_submissions   int   — from auth system if available
  ip_asn_class        str   — "residential" | "datacenter" | "vpn" | "tor"
  reporter_relation   str   — "witness" | "second_hand" | "anonymous"
  device_hash         str   — used by rate_limit_check_node
  browser_fp_hash     str

Inputs (from ExtractionResult, already in state after validate_node):
  soc_hash            str   — hash of name+org (set by validate_node via hash_soc())

ES lookup for named reporters:
  queries confirmed_hoax_count and confirmed_real_count from indexed documents
  that have analyst_decision set.

Credibility formula:
  anonymous:  base = 0.4  (can't verify identity or history)
  named:      base = 0.5
    + 0.10 × min(prior_law_enforcement_contacts, 3)  — known to LE = credible
    - 0.10  if acct_age_days < 7                     — brand-new account
    - 0.12  if ip_asn_class in datacenter/vpn/tor     — suspicious network
    + 0.15  if confirmed_real_ratio > 0.7             — proven trustworthy history
    - 0.15  × confirmed_hoax_prior_reports            — penalise each confirmed hoax
    + 0.08  × confirmed_real_prior_reports            — reward confirmed real reports
  clip to [0.1, 1.0]
"""
from __future__ import annotations

from elasticsearch import Elasticsearch

from langraph_app.config.settings import get_settings


_SUSPICIOUS_ASN_CLASSES = {"datacenter", "vpn", "tor"}


def _get_es() -> Elasticsearch:
    s = get_settings()
    kwargs: dict = {"hosts": [s.es_host]}
    if s.es_api_key:
        kwargs["api_key"] = s.es_api_key
    return Elasticsearch(**kwargs)


def _lookup_reporter_history(es: Elasticsearch, index: str, soc_hash: str) -> dict:
    """
    Query ES for prior reports from this reporter (matched by soc_hash).
    Returns confirmed_hoax_count, confirmed_real_count.
    """
    if not soc_hash:
        return {"confirmed_hoax_count": 0, "confirmed_real_count": 0}
    try:
        resp = es.search(
            index=index,
            body={
                "query": {
                    "term": {"metadata.soc_hash": soc_hash}
                },
                "_source": ["metadata.analyst_decision"],
                "size": 50,
            },
        )
        hoax_count = 0
        real_count = 0
        for hit in resp.get("hits", {}).get("hits", []):
            decision = (
                hit.get("_source", {})
                .get("metadata", {})
                .get("analyst_decision", "")
            )
            if decision == "hoax":
                hoax_count += 1
            elif decision == "real":
                real_count += 1
        return {"confirmed_hoax_count": hoax_count, "confirmed_real_count": real_count}
    except Exception:
        return {"confirmed_hoax_count": 0, "confirmed_real_count": 0}


def _score_reporter(
    reporter: dict,
    soc_history: dict,
    confirmed_hoax_count: int,
    confirmed_real_count: int,
) -> tuple[float, bool, dict]:
    """
    Compute credibility score and breakdown.
    Returns (score, is_anonymous, breakdown_dict).
    """
    is_anonymous = bool(reporter.get("anonymous", True))
    acct_age_days = int(reporter.get("acct_age_days", 0))
    ip_asn_class = reporter.get("ip_asn_class", "residential").lower()
    prior_law_enforcement_contacts = int(
        soc_history.get("prior_law_enforcement_contacts", 0)
    )

    breakdown: dict = {}

    if is_anonymous:
        score = 0.4
        breakdown["base"] = 0.4
        breakdown["reason"] = "anonymous — identity unverifiable"
    else:
        score = 0.5
        breakdown["base"] = 0.5

        # Law-enforcement contact bonus
        le_bonus = 0.10 * min(prior_law_enforcement_contacts, 3)
        if le_bonus:
            score += le_bonus
            breakdown["le_contact_bonus"] = le_bonus

        # Brand-new account penalty
        if acct_age_days < 7:
            score -= 0.10
            breakdown["new_account_penalty"] = -0.10

        # Suspicious network penalty
        if ip_asn_class in _SUSPICIOUS_ASN_CLASSES:
            score -= 0.12
            breakdown["suspicious_network_penalty"] = -0.12
            breakdown["ip_asn_class"] = ip_asn_class

        # History adjustments
        if confirmed_hoax_count > 0:
            hoax_penalty = 0.15 * confirmed_hoax_count
            score -= hoax_penalty
            breakdown["confirmed_hoax_penalty"] = -hoax_penalty

        if confirmed_real_count > 0:
            real_bonus = 0.08 * confirmed_real_count
            # High real-accuracy bonus
            total = confirmed_hoax_count + confirmed_real_count
            real_ratio = confirmed_real_count / total if total > 0 else 0.0
            if real_ratio > 0.7:
                real_bonus += 0.15
                breakdown["high_accuracy_bonus"] = 0.15
            score += real_bonus
            breakdown["confirmed_real_bonus"] = real_bonus

    # Clip to valid range
    score = max(0.1, min(1.0, score))
    breakdown["final_score"] = score

    return score, is_anonymous, breakdown


def run(state: dict) -> dict:
    if state.get("guardrail_hard_block"):
        return {}

    s = get_settings()
    raw_report: dict = state.get("raw_report") or {}
    reporter: dict = raw_report.get("reporter", {})
    soc_history: dict = (
        (state.get("extraction_result") or {})
        .get("soc_history", {})
    )
    soc_hash: str = state.get("soc_hash", "")

    # ES history lookup for named reporters only
    confirmed_hoax_count = 0
    confirmed_real_count = 0
    if soc_hash and not reporter.get("anonymous", True):
        try:
            es = _get_es()
            history = _lookup_reporter_history(es, s.es_index, soc_hash)
            confirmed_hoax_count = history["confirmed_hoax_count"]
            confirmed_real_count = history["confirmed_real_count"]
        except Exception:
            pass  # non-fatal — proceed with base score

    score, is_anonymous, breakdown = _score_reporter(
        reporter,
        soc_history,
        confirmed_hoax_count,
        confirmed_real_count,
    )

    breakdown["confirmed_hoax_prior_reports"] = confirmed_hoax_count
    breakdown["confirmed_real_prior_reports"] = confirmed_real_count

    return {
        "reporter_credibility_score": score,
        "is_anonymous_reporter": is_anonymous,
        "credibility_breakdown": breakdown,
    }
