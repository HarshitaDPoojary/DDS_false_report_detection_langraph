"""
Incident Intake Agent — FastAPI + OpenAI (GPT-4o)

Accepts: report_id, report (free text), attachments (images/PDFs)
Returns: strict JSON matching your schema.

Setup
-----
pip install fastapi uvicorn pydantic python-multipart openai pillow pdf2image pytesseract
# optional system deps for pdf2image & OCR:
# macOS:  brew install poppler tesseract
# Ubuntu: sudo apt-get install -y poppler-utils tesseract-ocr

ENV:
  OPENAI_API_KEY=sk-...
  SOC_SALT=your-pepper (optional)
"""

from __future__ import annotations
import io, os, re, json, base64, hashlib, datetime as dt
from typing import List, Optional
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# Optional image/PDF OCR helpers
try:
    from PIL import Image
except Exception:
    Image = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    import pytesseract
except Exception:
    pytesseract = None

# ---------- Schema (your exact fields) ----------

INCIDENT_TYPES = {
    "bomb_threat","gun_threat","fight","burglary","arson","assault",
    "sexual_assault","stalking","vandalism","extortion","doxxing","other"
}
FIRST_SECOND = {"first_hand","second_hand","unknown"}

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

class SOC_History(BaseModel):
    prior_reports: int = 0
    restraining_or_protection_order: bool = False
    prior_law_enforcement_contacts: int = 0

class GrievanceContext(BaseModel):
    event: str = "unknown"  # suspension|breakup|firing|unknown
    days_since: int = 0

class ExtractionResult(BaseModel):
    incident_type: str = "other"
    who: Who = Field(default_factory=Who)
    where: Where = Field(default_factory=Where)
    when_window: WhenWindow = Field(default_factory=WhenWindow)
    means: Means = Field(default_factory=Means)
    first_second_hand: str = "unknown"
    targets: List[str] = []

    # Subject of Concern (SoC)
    soc_key: str = ""
    soc_history: SOC_History = Field(default_factory=SOC_History)
    grievance_context: GrievanceContext = Field(default_factory=GrievanceContext)

    # Evidence
    quotes: List[str] = []
    screens_evidence: bool = False
    named_items: List[str] = []
    vehicle: Vehicle = Field(default_factory=Vehicle)

    # Provenance / debug (optional)
    report_id: str = ""
    notes: List[str] = []

# ---------- Config ----------

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # vision + JSON mode
SOC_SALT = os.environ.get("SOC_SALT", "pepper")
client = OpenAI()

# ---------- Helpers ----------

def hash_soc(name: str, org: str, dob_fragment: str = "") -> str:
    h = hashlib.sha256()
    h.update((SOC_SALT + "|" + name.strip().lower() + "|" + org.strip().lower() + "|" + dob_fragment.strip()).encode("utf-8"))
    return h.hexdigest()[:24]

def to_data_url(img_bytes: bytes, fallback_mime: str = "image/png") -> str:
    # crude sniff
    magic = img_bytes[:4]
    mime = fallback_mime
    if magic.startswith(b"\xff\xd8"): mime = "image/jpeg"
    elif magic.startswith(b"\x89PNG"): mime = "image/png"
    elif magic.startswith(b"GIF8"):    mime = "image/gif"
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def pil_bytes_from_pdf(pdf_bytes: bytes) -> List[bytes]:
    """Convert PDF pages to PNG bytes for vision models (no OCR needed)."""
    if convert_from_bytes is None or Image is None:
        return []
    images = convert_from_bytes(pdf_bytes)  # list of PIL images
    out = []
    for im in images:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out

def ocr_image_bytes(img_bytes: bytes) -> str:
    """Optional OCR to add text context; GPT-4o can see image directly, this is bonus evidence text."""
    if pytesseract is None or Image is None:
        return ""
    try:
        im = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(im)
    except Exception:
        return ""

def quick_quotes(text: str) -> List[str]:
    quotes = []
    for l, r in [("'", "'"), ('"', '"'), ("“","”"), ("‘","’")]:
        pattern = re.escape(l) + r"(.+?)" + re.escape(r)
        for m in re.finditer(pattern, text, re.S):
            q = m.group(1).strip()
            if 3 <= len(q) <= 400: quotes.append(q)
    # threat-like fallback
    for m in re.finditer(r"\b(will|gonna|going\s+to)\s+(kill|shoot|detonate|burn|hurt)\b.*", text, re.I):
        quotes.append(m.group(0)[:200])
    # dedupe preserving order
    seen, out = set(), []
    for q in quotes:
        if q not in seen:
            seen.add(q); out.append(q)
    return out

# ---------- LLM call ----------

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
  "vehicle": {"plate": "", "make_model": "", "color": ""}
}
Rules:
- Do not invent facts. If unknown, leave empty string/array/false (or [] for geo).
- Extract quotes verbatim from text or images.
"""

def call_gpt(report_id: str, text_blocks: List[str], image_data_urls: List[str]) -> dict:
    """
    Send mixed text + images to GPT-4o, ask for strict JSON back.
    """
    messages = [
        {"role": "system", "content": "You are a strict JSON extraction engine for incident reports."},
        {"role": "user", "content": [{"type": "text", "text": f"Report ID: {report_id}\n\n{SCHEMA_HINT}"}]}
    ]

    # Add main report text (first)
    if text_blocks:
        messages.append({"role": "user", "content": [{"type": "text", "text": "REPORT TEXT:\n" + "\n\n".join(text_blocks)}]})

    # Add each image as a vision input
    for data_url in image_data_urls:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": "EVIDENCE IMAGE:"},
            {"type": "input_image", "image_url": {"url": data_url}}
        ]})

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except Exception:
        return {}

# ---------- API ----------

app = FastAPI(title="Incident Intake Agent — OpenAI (GPT-4o)")

@app.post("/v1/extract")
async def extract(
    report_id: str = Form(...),
    report: str = Form(""),
    attachments: Optional[List[UploadFile]] = None,
):
    # Collect text & images for GPT
    text_blocks: List[str] = []
    image_data_urls: List[str] = []
    screens_evidence = False

    if report.strip():
        text_blocks.append(report.strip())

    if attachments:
        for f in attachments:
            name = (f.filename or "").lower()
            try:
                content = await f.read()
            except Exception:
                continue

            if name.endswith((".png",".jpg",".jpeg",".webp",".gif",".bmp")):
                # send to GPT as image; also optional OCR into text for extra context
                image_data_urls.append(to_data_url(content))
                ocr_txt = ocr_image_bytes(content)
                if ocr_txt.strip():
                    text_blocks.append("[OCR FROM IMAGE]\n" + ocr_txt.strip())
                screens_evidence = True

            elif name.endswith(".pdf"):
                # Convert each page to image and send to GPT (best), + optional OCR text
                if convert_from_bytes and Image:
                    page_imgs = pil_bytes_from_pdf(content)
                    for page in page_imgs[:6]:  # safety cap first 6 pages
                        image_data_urls.append(to_data_url(page))
                        if pytesseract:
                            t = ocr_image_bytes(page)
                            if t.strip():
                                text_blocks.append("[OCR FROM PDF PAGE]\n" + t.strip())
                    screens_evidence = screens_evidence or len(page_imgs) > 0
                else:
                    # fallback: no vision page conversion; try OCR full PDF (if tesseract exists)
                    if pytesseract and Image:
                        # best-effort OCR by rasterizing via pdf2image—already covered above
                        pass

            else:
                # (Optional) add audio → Whisper, docs → different pipeline
                pass

    # Quick local quotes to improve recall (merge later)
    local_quotes = quick_quotes("\n\n".join(text_blocks)) if text_blocks else []

    # LLM extraction
    llm_json = call_gpt(report_id, text_blocks, image_data_urls)

    # Merge with defaults & validate
    defaults = ExtractionResult().model_dump()
    merged = {**defaults, **llm_json}

    # enum sanity
    if merged.get("incident_type") not in INCIDENT_TYPES:
        merged["incident_type"] = "other"
    if merged.get("first_second_hand") not in FIRST_SECOND:
        merged["first_second_hand"] = "unknown"

    # nested presence
    merged.setdefault("who", {})
    merged.setdefault("where", {})
    merged.setdefault("when_window", {})
    merged.setdefault("means", {})
    merged.setdefault("vehicle", {})
    merged.setdefault("soc_history", {"prior_reports": 0, "restraining_or_protection_order": False, "prior_law_enforcement_contacts": 0})
    merged.setdefault("grievance_context", {"event":"unknown", "days_since":0})

    # screens_evidence: if we attached images/pdf pages or model flagged it
    merged["screens_evidence"] = bool(merged.get("screens_evidence") or screens_evidence)

    # quotes: union of model + local
    q = merged.get("quotes") or []
    if not isinstance(q, list): q = []
    # dedupe
    seen, uq = set(), []
    for s in q + local_quotes:
        if s not in seen:
            seen.add(s); uq.append(s)
    merged["quotes"] = uq

    # soc_key if missing
    who = merged.get("who", {})
    names = who.get("named_persons", []) or []
    org = who.get("target_org", "") or ""
    if not merged.get("soc_key") and names:
        merged["soc_key"] = hash_soc(names[0], org)

    # provenance
    merged["report_id"] = report_id
    merged.setdefault("notes", []).append("extracted via GPT-4o")

    # final validation
    try:
        result = ExtractionResult(**merged)
        return JSONResponse(status_code=200, content={"ok": True, **result.model_dump()})
    except ValidationError as e:
        return JSONResponse(status_code=200, content={
            "ok": False,
            "report_id": report_id,
            "errors": json.loads(e.json()),
            "partial": merged
        })
