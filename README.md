# False Report Identification System

A LangGraph-based pipeline for automated triage of law enforcement incident reports. Accepts free-text reports and optional attachments, runs multi-stage AI analysis, and returns a structured risk assessment with hoax probability scoring.

## Architecture Overview

Three LangGraph pipelines run in sequence per submission:

```
Submission (POST /v1/reports)
        │
        ▼
  IntakeGraph          ← guardrails, OCR, vision, attachment classification + extraction, LLM structured extraction
        │
        ▼
  IngestionGraph       ← embed text, classify incident type, index to Elasticsearch
        │
        ▼
  AnalysisGraph        ← retrieve similar reports, score credibility, hoax detection, final risk assessment
        │
        ▼
  MongoDB (status=complete)  ← analyst polls GET /v1/reports/{id}
```

### IntakeGraph — Intelligent Attachment Processing

The intake pipeline auto-classifies every attachment and routes it to a specialized extraction node:

```
guardrails_node
      │
check_attachments_node
      ├── no attachments → extract_node
      └── has attachments → classify_attachments_node
                                  │ (parallel fan-out)
            ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
        screenshot  vehicle  id_document  person    ocr      vision   image_meta
          _node      _node      _node      _node    _node    _node      _node
            └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
                                  │ (fan-in)
                            extract_node
                                  │
                            validate_node → END
```

**Attachment types and what each node extracts:**

| Type | Node | Output |
|------|------|--------|
| `screenshot` | `screenshot_node` | Structured chat transcript: `[{sender, message, timestamp, platform}]` |
| `vehicle` | `vehicle_node` | `{plate, make_model, color, damage_description}` |
| `id_document` | `id_document_node` | `{full_name, date_of_birth, address, id_number, issuer_state, expiry_date, document_type}` |
| `person` | `person_node` | `{appearance, is_suspect, face_match_result}` |
| `document` / `unknown` | `ocr_node` + `vision_node` | Raw OCR text + visual description |

Classification uses two phases: filename keyword heuristics first (fast, no API call), then a vLLM single-token prompt if the filename is ambiguous.

### Vision Model Backend

All image analysis uses a locally-hosted vLLM server (Qwen2-VL-7B) with automatic fallback to NVIDIA NIM:

1. **Local vLLM** — probed at startup via `/health` endpoint (3s timeout). Used if reachable.
2. **NVIDIA NIM** — `nvidia/llama-3.2-90b-vision-instruct` via API. Used if local vLLM is unreachable.
3. **Error** — raised if neither backend is configured.

Qwen2-VL is used instead of GPT-4o because GPT-4o refuses to read license plate text and government ID numbers.

### Face Recognition (Optional)

When a person attachment is submitted in a suspect context (report text contains "suspect" or `form_metadata.suspect_photo=true`), InsightFace embeds the face and cosine-searches a known-offender JSON database.

Face recognition only activates when **all three** conditions are met:
- `FACE_RECOGNITION_ENABLED=true`
- `KNOWN_OFFENDER_DB_PATH` points to a valid DB file
- `insightface` library is installed

The system works without InsightFace — face recognition degrades gracefully to appearance description only.

## File Structure

```
langraph_app/
├── api/
│   └── server.py               # FastAPI endpoints + background pipeline runner
├── config/
│   └── settings.py             # Pydantic-settings config (all values from .env)
├── db/
│   └── mongo.py                # MongoDB helpers (insert, mark_complete, get_report, etc.)
├── graphs/
│   ├── intake_graph.py         # IntakeGraph wiring (classify + fan-out + fan-in)
│   ├── ingestion_graph.py      # IngestionGraph (embed + classify + ES index)
│   └── analysis_graph.py       # AnalysisGraph (retrieve + score + hoax + final)
├── models/
│   └── report.py               # Pydantic models: ExtractionResult, ChatMessage, IDDocument, PersonDescription, Vehicle
├── nodes/
│   ├── guardrails.py           # PII redaction, injection detection, hard-block logic
│   ├── check_attachments.py    # Sets has_attachments flag
│   ├── classify_attachments.py # Two-phase attachment type classification
│   ├── screenshot_node.py      # Chat transcript extraction from screenshots
│   ├── vehicle_node.py         # Plate + make/model + damage extraction
│   ├── id_document_node.py     # Identity field extraction from IDs/passports
│   ├── person_node.py          # Appearance description + optional InsightFace match
│   ├── ocr.py                  # pytesseract / SmolDocling PDF OCR
│   ├── vision.py               # vLLM visual description for unclassified images
│   ├── image_metadata.py       # EXIF extraction + GPS conflict detection
│   ├── extract.py              # LLM structured extraction → ExtractionResult
│   └── validate.py             # Pydantic validation + SOC hash + quote extraction
├── utils/
│   ├── vllm_client.py          # vLLM/NIM ChatOpenAI client with health-check fallback
│   ├── image_utils.py          # to_data_url, ocr_image_bytes, pil_bytes_from_pdf
│   └── text_utils.py           # hash_soc, quick_quotes
└── state.py                    # IntakeState TypedDict

tests/
├── test_vllm_integration.py    # 33 tests for vLLM client + all new nodes + intake graph smoke test
├── test_intake_graph.py
├── test_ingestion_graph.py
├── test_analysis_graph.py
└── test_nodes.py

legacy/                         # Pre-LangGraph scripts (reference only, not used by pipeline)
```

## Installation

```bash
# Activate the project virtualenv
test\Scripts\activate   # Windows
# source test/bin/activate  # Linux/macOS

# Core dependencies
pip install langchain-openai langgraph pydantic-settings fastapi uvicorn pymongo elasticsearch

# OCR (optional — used for unclassified images and PDF fallback)
pip install pytesseract pillow pdf2image

# Face recognition (optional — only needed if FACE_RECOGNITION_ENABLED=true)
pip install insightface onnxruntime opencv-python-headless
```

## Configuration

Copy `.env.example` to `.env` and fill in values:

```env
# OpenAI (used by extract_node for structured extraction)
OPENAI_API_KEY=sk-...

# vLLM — locally hosted Qwen2-VL-7B (requires ~16 GB VRAM)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=token-abc123
VLLM_VISION_MODEL=Qwen/Qwen2-VL-7B-Instruct

# NVIDIA NIM fallback (used automatically if vLLM is unreachable)
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NIM_API_KEY=nvapi-...
NIM_VISION_MODEL=nvidia/llama-3.2-90b-vision-instruct

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB=incident_reports

# Elasticsearch
ES_HOST=https://your-es-host:9200
ES_API_KEY=your-es-api-key

# Security
SOC_SALT=your-random-pepper-string
API_KEY=your-analyst-api-key

# Face recognition (optional)
FACE_RECOGNITION_ENABLED=false
KNOWN_OFFENDER_DB_PATH=/path/to/offenders.json
FACE_MATCH_THRESHOLD=0.60
```

**Known-offender DB format** (`KNOWN_OFFENDER_DB_PATH`):
```json
[
  {"offender_id": "OFF-001", "name": "John Doe", "embedding": [0.123, -0.456, ...]}
]
```
Each `embedding` must be a 512-float ArcFace normalized embedding.

## Running

```bash
# Start the API server
uvicorn langraph_app.api.server:app --host 0.0.0.0 --port 8080

# Submit a report with attachments
curl -X POST http://localhost:8080/v1/reports \
  -F "report=Suspect was seen fleeing in a blue sedan near the school." \
  -F "attachments=@/path/to/screenshot.jpg" \
  -F "attachments=@/path/to/car_photo.jpg"

# Poll for result
curl -H "X-API-Key: your-analyst-api-key" \
  http://localhost:8080/v1/reports/{report_id}
```

## API Reference

### `POST /v1/reports`
Submit a new incident report. Returns immediately with `report_id`; AI analysis runs in the background.

**Form fields:**
- `report` (required) — free-text incident description
- `attachments` (optional, repeatable) — image or PDF files
- `reporter_name` (optional) — reporter's name
- `report_id` (optional) — custom UUID; generated if omitted

**Rate limit:** 10 requests/minute per IP.

**Response:**
```json
{"report_id": "uuid", "status": "submitted"}
```

### `GET /v1/reports/{report_id}`
Poll analysis status or retrieve completed result. Requires `X-API-Key` header.

**Response when pending:**
```json
{"status": "submitted"}
```

**Response when complete:**
```json
{
  "status": "complete",
  "incident_type": "theft",
  "hoax_probability": 0.12,
  "risk_level": "medium",
  "chat_transcript": [...],
  "vehicle_extractions": [...],
  "id_document_extractions": [...],
  "person_descriptions": [...],
  "attachment_types": ["screenshot", "vehicle"],
  ...
}
```

### `POST /v1/feedback`
Record analyst decision. Requires `X-API-Key` header.

```json
{
  "report_id": "uuid",
  "analyst_decision": "real",
  "analyst_notes": "Verified with CCTV",
  "decided_by": "analyst@agency.gov"
}
```

`analyst_decision` must be one of: `real`, `hoax`, `inconclusive`.

### `PATCH /v1/reports/{report_id}`
Toggle data retention flag. Requires `X-API-Key` header.

```json
{"retain_indefinitely": true}
```

### `GET /health`
Liveness probe — returns `{"status": "ok"}`.

## Testing

```bash
# Unit + integration tests (no GPU or live backends required — all LLM calls mocked)
test\Scripts\python.exe -m pytest tests/test_vllm_integration.py -v

# Full test suite
test\Scripts\python.exe -m pytest tests/ -v

# GPU VRAM check (advisory — does not affect test results)
test\Scripts\python.exe -c "from langraph_app.utils.vllm_client import check_gpu_vram; print(check_gpu_vram())"
```

`tests/test_vllm_integration.py` covers:
- vLLM client: local-first health check, NIM fallback, error when neither configured
- `classify_attachments`: filename heuristics, vLLM fallback, hard-block guard
- `screenshot_node`: JSON transcript extraction, pytesseract fallback, hard-block guard
- `vehicle_node`: plate/make/model extraction, normalization, hard-block guard
- `id_document_node`: identity field extraction, document type normalization
- `person_node`: appearance description, suspect detection, InsightFace mock
- `extract_node`: structured field injection, post-invoke overwrite of LLM output
- Full intake graph smoke test: all 4 attachment types in one submission

## vLLM Setup (Local GPU)

Qwen2-VL-7B requires approximately 16 GB VRAM. A single RTX 3090/4090 or A10G is sufficient.

```bash
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key token-abc123 \
  --max-model-len 4096

# Verify health
curl http://localhost:8000/health
```

If VRAM is insufficient, set only `NIM_API_KEY` and `NIM_BASE_URL` in `.env` — the pipeline will use NVIDIA NIM automatically.
