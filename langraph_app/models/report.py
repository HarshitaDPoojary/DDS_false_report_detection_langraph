"""
Pydantic models for structured incident report extraction.
Migrated from legacy/gpt_incident_agent.py — schema unchanged.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


INCIDENT_TYPES = {
    "bomb_threat", "gun_threat", "fight", "burglary", "arson", "assault",
    "sexual_assault", "stalking", "vandalism", "extortion", "doxxing", "other"
}
FIRST_SECOND = {"first_hand", "second_hand", "unknown"}


class Who(BaseModel):
    named_persons: List[str] = []
    aliases: List[str] = []
    target_org: str = ""


class Where(BaseModel):
    venue: str = ""
    address: str = ""
    room: str = ""
    entrance: str = ""
    geo: List[float] = []  # [lat, lng]


class WhenWindow(BaseModel):
    start_iso: str = ""
    end_iso: str = ""


class Means(BaseModel):
    weapon: str = ""
    materials: str = ""
    method: str = ""


class Vehicle(BaseModel):
    plate: str = ""
    make_model: str = ""
    color: str = ""
    damage_description: str = ""


class ChatMessage(BaseModel):
    sender: str = ""
    message: str = ""
    timestamp: str = ""
    platform: str = ""   # WhatsApp | SMS | Signal | Telegram | iMessage | Email | other


class IDDocument(BaseModel):
    full_name: str = ""
    date_of_birth: str = ""
    address: str = ""
    id_number: str = ""
    issuer_state: str = ""
    expiry_date: str = ""
    document_type: str = ""  # drivers_license | passport | state_id | unknown


class PersonDescription(BaseModel):
    appearance: str = ""
    is_suspect: bool = False
    face_match_result: Dict = Field(default_factory=dict)
    # face_match_result keys: matched (bool), offender_id (str), name (str), similarity (float)


class SOC_History(BaseModel):
    prior_reports: int = 0
    restraining_or_protection_order: bool = False
    prior_law_enforcement_contacts: int = 0


class GrievanceContext(BaseModel):
    event: str = "unknown"   # suspension | breakup | firing | unknown
    days_since: int = 0


class ExtractionResult(BaseModel):
    incident_type: str = "other"
    who: Who = Field(default_factory=Who)
    where: Where = Field(default_factory=Where)
    when_window: WhenWindow = Field(default_factory=WhenWindow)
    means: Means = Field(default_factory=Means)
    first_second_hand: str = "unknown"
    targets: List[str] = []

    # Subject of Concern
    soc_key: str = ""
    soc_history: SOC_History = Field(default_factory=SOC_History)
    grievance_context: GrievanceContext = Field(default_factory=GrievanceContext)

    # Evidence
    quotes: List[str] = []
    screens_evidence: bool = False
    named_items: List[str] = []
    vehicle: Vehicle = Field(default_factory=Vehicle)

    # Provenance
    report_id: str = ""
    notes: List[str] = []

    # Attachment classification + structured extraction (populated by specialized nodes)
    attachment_types: List[str] = []
    chat_transcript: List[ChatMessage] = Field(default_factory=list)
    id_document: IDDocument = Field(default_factory=IDDocument)
    person_descriptions: List[PersonDescription] = Field(default_factory=list)


# Schema hint string sent to GPT as part of the extraction prompt
SCHEMA_HINT = """
Return ONLY valid JSON matching this schema:
{
  "incident_type": "bomb_threat|gun_threat|fight|burglary|arson|assault|sexual_assault|stalking|vandalism|extortion|doxxing|other",
  "who": {"named_persons": [], "aliases": [], "target_org": ""},
  "where": {"venue": "", "address": "", "room": "", "entrance": "", "geo": []},
  "when_window": {"start_iso": "", "end_iso": ""},
  "means": {"weapon": "", "materials": "", "method": ""},
  "first_second_hand": "first_hand|second_hand|unknown",
  "targets": [],
  "soc_key": "",
  "soc_history": {"prior_reports": 0, "restraining_or_protection_order": false, "prior_law_enforcement_contacts": 0},
  "grievance_context": {"event": "suspension|breakup|firing|unknown", "days_since": 0},
  "quotes": [],
  "screens_evidence": false,
  "named_items": [],
  "vehicle": {"plate": "", "make_model": "", "color": "", "damage_description": ""},
  "attachment_types": [],
  "chat_transcript": [],
  "id_document": {"full_name": "", "date_of_birth": "", "address": "", "id_number": "", "issuer_state": "", "expiry_date": "", "document_type": ""},
  "person_descriptions": []
}
Rules:
- Do not invent facts. If unknown, leave empty string/array/false/0.
- Extract threat quotes verbatim.
"""
