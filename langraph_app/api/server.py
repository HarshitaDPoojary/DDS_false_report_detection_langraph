"""
FastAPI server — report submission, analysis polling, feedback, and health check.

Endpoints:
  POST /v1/reports          — submit a new report (text + optional attachments)
  GET  /v1/reports/{id}     — poll analysis status / get full result
  PATCH /v1/reports/{id}    — toggle retain_indefinitely flag (analyst only)
  POST /v1/feedback         — record analyst decision (analyst only)
  GET  /health              — liveness probe

Submission flow:
  1. FastAPI receives form data + optional file uploads
  2. Collects reporter metadata from HTTP headers
  3. Writes initial MongoDB document: status="submitted"
  4. Returns 201 { report_id, status: "submitted" }  ← no waiting for AI
  5. BackgroundTasks fires run_pipeline() which:
       a. IntakeGraph    — guardrails + OCR + vision + EXIF + LLM extraction
       b. IngestionGraph — embed + classify + index to Elasticsearch
       c. AnalysisGraph  — guardrails + retrieve + score + hoax + final
       d. Updates MongoDB with status="complete" or status="failed"

Security:
  - Public endpoints (POST /v1/reports, GET /health): rate-limited, no auth required
  - Analyst endpoints (GET/PATCH /v1/reports/{id}, POST /v1/feedback): require API key
  - Per-IP rate limiting via slowapi: 10 POST /v1/reports per minute
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from langraph_app.config.settings import get_settings
from langraph_app.db import mongo
from langraph_app.graphs.intake_graph import intake_graph
from langraph_app.graphs.ingestion_graph import ingestion_graph
from langraph_app.graphs.analysis_graph import analysis_graph


# ---------------------------------------------------------------------------
# App + rate limiter
# ---------------------------------------------------------------------------

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="False Report Identification API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """Require X-API-Key header for analyst-facing endpoints."""
    s = get_settings()
    if not s.api_key:
        return  # API key auth disabled (dev mode)
    if x_api_key != s.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Reporter metadata helpers
# ---------------------------------------------------------------------------

def _collect_reporter_meta(request: Request, reporter_name: str) -> dict:
    """Build reporter metadata dict from HTTP request headers."""
    user_agent = request.headers.get("User-Agent", "")
    device_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:16]
    browser_fp = request.headers.get("X-Browser-FP", "")
    ip = request.client.host if request.client else ""

    # Classify IP ASN class — simplified heuristic; replace with MaxMind in prod
    ip_asn_class = "residential"
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        ip = forwarded.split(",")[0].strip()

    return {
        "anonymous": not bool(reporter_name),
        "device_hash": device_hash,
        "browser_fp_hash": browser_fp,
        "ip_asn_class": ip_asn_class,
        "acct_age_days": 0,
        "prior_submissions": 0,
        "reporter_relation": "anonymous" if not reporter_name else "witness",
    }


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    report_id: str,
    raw_text: str,
    attachments: list[bytes],
    attachment_names: list[str],
    reporter_meta: dict,
) -> None:
    """
    Full LangGraph pipeline run as a FastAPI background task.
    Outcomes (success and failure) are written to MongoDB — frontend polls GET.
    """
    try:
        # ── Step 1: IntakeGraph ──────────────────────────────────────────────
        intake_state = {
            "report_id": report_id,
            "text_input": raw_text,
            "attachments": attachments,
            "attachment_names": attachment_names,
            "form_metadata": {"reporter": reporter_meta},
            "has_attachments": len(attachments) > 0,
        }
        intake_result = await intake_graph.ainvoke(intake_state)

        if intake_result.get("guardrail_hard_block"):
            mongo.mark_failed(report_id, intake_result.get("error", "Guardrail block"))
            return

        validated_report: dict = intake_result.get("validated_report") or {}

        # ── Step 2: IngestionGraph ───────────────────────────────────────────
        ingestion_state = {
            "report_id": report_id,
            "raw_report": validated_report,
            "free_text": intake_result.get("free_text", raw_text),
            "location": intake_result.get("location", []),
            "time_start": intake_result.get("time_start", ""),
            "time_end": intake_result.get("time_end", ""),
            "time_midpoint": intake_result.get("time_midpoint", ""),
            "text_embedding": intake_result.get("text_embedding", []),
            "incident_types": intake_result.get("incident_types", []),
            "severity": intake_result.get("severity", "low"),
            "validated_report": validated_report,
            "image_metadata": intake_result.get("image_metadata", []),
            "visual_description": intake_result.get("visual_description", ""),
        }
        ingestion_result = await ingestion_graph.ainvoke(ingestion_state)

        if ingestion_result.get("error"):
            mongo.mark_failed(report_id, ingestion_result["error"])
            return

        # ── Step 3: AnalysisGraph ────────────────────────────────────────────
        analysis_state = {
            "report_id": report_id,
            "raw_report": {**validated_report, "reporter": reporter_meta},
            "free_text": ingestion_state["free_text"],
            "location": ingestion_state["location"],
            "time_start": ingestion_state["time_start"],
            "time_end": ingestion_state["time_end"],
            "time_midpoint": ingestion_state["time_midpoint"],
            "text_embedding": ingestion_result.get("text_embedding") or ingestion_state["text_embedding"],
            "incident_types": ingestion_result.get("incident_types") or ingestion_state["incident_types"],
            "severity": ingestion_result.get("severity") or ingestion_state["severity"],
            "image_metadata": intake_result.get("image_metadata", []),
            "image_metadata_conflicts": intake_result.get("image_metadata_conflicts", []),
            "visual_description": intake_result.get("visual_description", ""),
            "soc_hash": intake_result.get("soc_hash", ""),
            "extraction_result": intake_result.get("extraction_result", {}),
            "radius_miles": 5.0,
            "lookback_hours": 24.0,
            "scorer_weights": {},
        }
        analysis_result = await analysis_graph.ainvoke(analysis_state)

        if analysis_result.get("guardrail_hard_block"):
            mongo.mark_failed(report_id, analysis_result.get("error", "Guardrail block"))
            return

        # ── Step 4: Persist complete result to MongoDB ───────────────────────
        final = analysis_result.get("analysis_result") or {}
        final["image_metadata"] = intake_result.get("image_metadata", [])
        final["visual_description"] = intake_result.get("visual_description", "")
        final["audit_trail"] = analysis_result.get("audit_trail", [])
        final["false_negative_risk"] = analysis_result.get("false_negative_risk", "low")
        final["escalation_reason"] = analysis_result.get("escalation_reason", "")

        mongo.mark_complete(report_id, final)

    except Exception as exc:
        mongo.mark_failed(report_id, str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/reports", status_code=201)
@limiter.limit("10/minute")
async def submit_report(
    request: Request,
    background_tasks: BackgroundTasks,
    report: str = Form(..., description="Report text"),
    report_id: Optional[str] = Form(default=None),
    reporter_name: Optional[str] = Form(default=None),
    attachments: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    """Submit a new incident report. Returns immediately; analysis runs in background."""
    if report_id is None:
        report_id = str(uuid.uuid4())

    # Read attachment bytes
    attachment_bytes: list[bytes] = []
    attachment_names: list[str] = []
    for f in attachments:
        data = await f.read()
        attachment_bytes.append(data)
        attachment_names.append(f.filename or "unknown")

    reporter_meta = _collect_reporter_meta(request, reporter_name or "")

    # Write initial record to MongoDB before returning
    mongo.insert_submitted(
        report_id=report_id,
        raw_text=report,
        attachments_count=len(attachment_bytes),
        reporter_meta=reporter_meta,
    )

    # Fire background pipeline — all outcomes written to MongoDB
    background_tasks.add_task(
        run_pipeline,
        report_id=report_id,
        raw_text=report,
        attachments=attachment_bytes,
        attachment_names=attachment_names,
        reporter_meta=reporter_meta,
    )

    return JSONResponse(
        status_code=201,
        content={"report_id": report_id, "status": "submitted"},
    )


@app.get("/v1/reports/{report_id}")
async def get_report(
    report_id: str,
    _: None = Depends(verify_api_key),
) -> JSONResponse:
    """Poll analysis status or retrieve full analysis result."""
    doc = mongo.get_report(report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return JSONResponse(content=doc)


class RetainRequest(BaseModel):
    retain_indefinitely: bool


@app.patch("/v1/reports/{report_id}")
async def patch_report(
    report_id: str,
    body: RetainRequest,
    _: None = Depends(verify_api_key),
) -> JSONResponse:
    """Toggle retain_indefinitely flag to stop or resume the TTL clock."""
    doc = mongo.get_report(report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Report not found")
    mongo.set_retain_indefinitely(report_id, body.retain_indefinitely)
    return JSONResponse(content={"status": "updated", "retain_indefinitely": body.retain_indefinitely})


class FeedbackRequest(BaseModel):
    report_id: str
    analyst_decision: str   # "real" | "hoax" | "inconclusive"
    analyst_notes: str = ""
    decided_by: str = ""


@app.post("/v1/feedback")
async def submit_feedback(
    body: FeedbackRequest,
    _: None = Depends(verify_api_key),
) -> JSONResponse:
    """Record analyst decision for a report."""
    allowed_decisions = {"real", "hoax", "inconclusive"}
    if body.analyst_decision not in allowed_decisions:
        raise HTTPException(
            status_code=422,
            detail=f"analyst_decision must be one of {sorted(allowed_decisions)}",
        )

    doc = mongo.get_report(body.report_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Report not found")

    mongo.record_feedback(
        report_id=body.report_id,
        analyst_decision=body.analyst_decision,
        analyst_notes=body.analyst_notes,
        decided_by=body.decided_by,
    )

    return JSONResponse(content={"status": "recorded"})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})
