"""
MongoDB client and report lifecycle helpers.

Collections:
  reports            — one document per submitted report; fields grow over lifecycle:
                       submitted → analyzed → decided
  analyst_decisions  — append-only audit log; one insert per POST /v1/feedback call

Report document lifecycle fields:
  report_id          str      — primary key
  raw_text           str      — original submission text
  attachments_count  int
  submitted_at       ISO str
  reporter_meta      dict     — from HTTP headers (device_hash, ip_asn_class, etc.)
  status             str      — "submitted" | "complete" | "failed"
  error              str|None — populated on status="failed"

  # Added after analysis completes:
  hoax_probability   float
  threat_level       float
  action             str
  confidence_range   list
  urgency_level      str
  ai_analysis        str
  cluster_summary    dict
  scored_results     list
  reporter_credibility_score float
  is_anonymous_reporter bool
  rate_limit_flagged bool
  image_metadata_conflicts list
  false_negative_risk str
  audit_trail        list

  # Added after analyst feedback:
  analyst_decision   str      — "real" | "hoax" | "inconclusive"
  analyst_notes      str
  decided_at         ISO str
  decided_by         str
  retain_indefinitely bool    — default False; set True to stop TTL clock

FeedbackRecord (analyst_decisions collection):
  report_id, submitted_at, hoax_probability, ai_analysis,
  analyst_decision, analyst_notes, decided_at, decided_by
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection

from langraph_app.config.settings import get_settings


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Optional[MongoClient] = None


def _get_client() -> MongoClient:
    global _client
    if _client is None:
        s = get_settings()
        _client = MongoClient(s.mongo_uri)
    return _client


def _reports() -> Collection:
    s = get_settings()
    return _get_client()[s.mongo_db]["reports"]


def _decisions() -> Collection:
    s = get_settings()
    return _get_client()[s.mongo_db]["analyst_decisions"]


# ---------------------------------------------------------------------------
# Report lifecycle helpers
# ---------------------------------------------------------------------------

def insert_submitted(
    report_id: str,
    raw_text: str,
    attachments_count: int,
    reporter_meta: dict,
) -> None:
    """Write initial report document with status='submitted'."""
    _reports().insert_one({
        "report_id": report_id,
        "raw_text": raw_text,
        "attachments_count": attachments_count,
        "reporter_meta": reporter_meta,
        "status": "submitted",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "retain_indefinitely": False,
    })


def mark_complete(report_id: str, analysis_result: dict) -> None:
    """Update report document with completed analysis results."""
    _reports().update_one(
        {"report_id": report_id},
        {"$set": {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in analysis_result.items()},
        }},
    )


def mark_failed(report_id: str, error: str) -> None:
    """Update report document with failure status and error message."""
    _reports().update_one(
        {"report_id": report_id},
        {"$set": {
            "status": "failed",
            "error": error,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }},
    )


def get_report(report_id: str) -> Optional[dict]:
    """Fetch report document by report_id. Returns None if not found."""
    doc = _reports().find_one({"report_id": report_id}, {"_id": 0})
    return doc


# ---------------------------------------------------------------------------
# Feedback / analyst decision helpers
# ---------------------------------------------------------------------------

def record_feedback(
    report_id: str,
    analyst_decision: str,
    analyst_notes: str,
    decided_by: str,
) -> None:
    """
    1. Update report document with analyst decision.
    2. Insert append-only audit record in analyst_decisions collection.
    Both writes happen in the same function — caller should handle exceptions.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Update the report document (single source of truth)
    _reports().update_one(
        {"report_id": report_id},
        {"$set": {
            "analyst_decision": analyst_decision,
            "analyst_notes": analyst_notes,
            "decided_at": now,
            "decided_by": decided_by,
        }},
    )

    # Fetch key fields for the audit record
    doc = _reports().find_one({"report_id": report_id}, {"_id": 0}) or {}

    # Append-only audit log — never updated after insert
    _decisions().insert_one({
        "report_id": report_id,
        "submitted_at": doc.get("submitted_at", ""),
        "hoax_probability": doc.get("hoax_probability"),
        "ai_analysis": doc.get("ai_analysis", ""),
        "analyst_decision": analyst_decision,
        "analyst_notes": analyst_notes,
        "decided_at": now,
        "decided_by": decided_by,
    })


def set_retain_indefinitely(report_id: str, retain: bool) -> None:
    """Allow analysts to stop (retain=True) or resume (retain=False) the TTL clock."""
    _reports().update_one(
        {"report_id": report_id},
        {"$set": {"retain_indefinitely": retain}},
    )
