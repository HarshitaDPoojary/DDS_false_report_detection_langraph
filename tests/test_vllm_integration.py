"""
Tests for vLLM integration and specialized attachment nodes.

All LLM calls are mocked — no GPU or live server required.

Coverage:
  vllm_client:           health check, local/NIM routing, GPU check
  classify_attachments:  filename heuristics, vLLM fallback, unknown
  screenshot_node:       transcript extraction, pytesseract fallback
  vehicle_node:          plate/make/model/color extraction
  id_document_node:      identity field extraction, doc type normalization
  person_node:           appearance description, suspect detection,
                         face recognition path (mocked), disabled path
  Full intake graph:     all 4 types in one submission
  Guardrail block:       all new nodes respect hard block
"""
from __future__ import annotations

import json
import unittest.mock as mock

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────────

def _jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff\xe0" + b"\x00" * 300


def _mock_llm_response(text: str):
    """Return a mock ChatOpenAI instance whose .invoke() returns text."""
    m = mock.MagicMock()
    m.invoke.return_value = mock.MagicMock(content=text)
    return m


def _mock_extraction_result() -> dict:
    return {
        "incident_type": "other",
        "who": {"named_persons": [], "aliases": [], "target_org": ""},
        "where": {"venue": "", "address": "", "room": "", "entrance": "", "geo": []},
        "when_window": {"start_iso": "", "end_iso": ""},
        "means": {"weapon": "", "materials": "", "method": ""},
        "first_second_hand": "unknown",
        "targets": [],
        "soc_key": "",
        "soc_history": {"prior_reports": 0, "restraining_or_protection_order": False,
                        "prior_law_enforcement_contacts": 0},
        "grievance_context": {"event": "unknown", "days_since": 0},
        "quotes": [],
        "screens_evidence": False,
        "named_items": [],
        "vehicle": {"plate": "", "make_model": "", "color": "", "damage_description": ""},
        "report_id": "",
        "notes": [],
        "attachment_types": [],
        "chat_transcript": [],
        "id_document": {},
        "person_descriptions": [],
    }


# ── vllm_client ───────────────────────────────────────────────────────────────

class TestVllmClient:
    def test_uses_local_when_healthy(self):
        from langraph_app.utils import vllm_client
        from langraph_app.config.settings import Settings

        settings = Settings(
            vllm_base_url="http://localhost:8000/v1",
            vllm_api_key="test-key",
            vllm_vision_model="Qwen/Qwen2-VL-7B-Instruct",
        )
        with (
            mock.patch("langraph_app.utils.vllm_client.get_settings", return_value=settings),
            mock.patch("langraph_app.utils.vllm_client._is_vllm_healthy", return_value=True),
            mock.patch("langraph_app.utils.vllm_client.ChatOpenAI") as mock_llm,
        ):
            vllm_client.get_vllm_client()
            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:8000/v1"
            assert call_kwargs["api_key"] == "test-key"

    def test_falls_back_to_nim_when_local_unreachable(self):
        from langraph_app.utils import vllm_client
        from langraph_app.config.settings import Settings

        settings = Settings(
            vllm_base_url="http://localhost:8000/v1",
            vllm_api_key="test-key",
            vllm_vision_model="Qwen/Qwen2-VL-7B-Instruct",
            nim_base_url="https://integrate.api.nvidia.com/v1",
            nim_api_key="nim-secret",
            nim_vision_model="nvidia/llama-3.2-90b-vision-instruct",
        )
        with (
            mock.patch("langraph_app.utils.vllm_client.get_settings", return_value=settings),
            mock.patch("langraph_app.utils.vllm_client._is_vllm_healthy", return_value=False),
            mock.patch("langraph_app.utils.vllm_client.ChatOpenAI") as mock_llm,
        ):
            vllm_client.get_vllm_client()
            call_kwargs = mock_llm.call_args.kwargs
            assert call_kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
            assert call_kwargs["api_key"] == "nim-secret"

    def test_raises_when_no_backend(self):
        from langraph_app.utils import vllm_client
        from langraph_app.config.settings import Settings

        settings = Settings(vllm_base_url="", nim_api_key="")
        with (
            mock.patch("langraph_app.utils.vllm_client.get_settings", return_value=settings),
            pytest.raises(RuntimeError, match="No vision backend available"),
        ):
            vllm_client.get_vllm_client()

    def test_health_check_strips_v1_suffix(self):
        from langraph_app.utils.vllm_client import _is_vllm_healthy
        with mock.patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = mock.Mock(return_value=False)
            mock_open.return_value.status = 200
            _is_vllm_healthy("http://localhost:8000/v1")
            called_url = mock_open.call_args[0][0]
            assert called_url == "http://localhost:8000/health"

    def test_gpu_check_returns_dict_keys(self):
        from langraph_app.utils.vllm_client import check_gpu_vram
        # Works even if nvidia-smi is absent
        result = check_gpu_vram()
        assert "total_mb" in result
        assert "free_mb" in result
        assert "sufficient_for_qwen2vl_7b" in result


# ── classify_attachments ──────────────────────────────────────────────────────

class TestClassifyAttachments:
    def test_screenshot_by_filename(self):
        from langraph_app.nodes.classify_attachments import run
        state = {
            "has_attachments": True,
            "attachments": [_jpeg_bytes()],
            "attachment_names": ["screenshot_chat.jpg"],
        }
        result = run(state)
        assert result["attachment_types"] == ["screenshot"]
        assert result["has_screenshot"] is True
        assert result["has_vehicle"] is False

    def test_vehicle_by_filename(self):
        from langraph_app.nodes.classify_attachments import run
        state = {
            "has_attachments": True,
            "attachments": [_jpeg_bytes()],
            "attachment_names": ["suspect_car.jpg"],
        }
        result = run(state)
        assert result["attachment_types"] == ["vehicle"]
        assert result["has_vehicle"] is True

    def test_id_document_by_filename(self):
        from langraph_app.nodes.classify_attachments import run
        state = {
            "has_attachments": True,
            "attachments": [_jpeg_bytes()],
            "attachment_names": ["drivers_license.jpg"],
        }
        result = run(state)
        assert result["attachment_types"] == ["id_document"]
        assert result["has_id_document"] is True

    def test_person_by_filename(self):
        from langraph_app.nodes.classify_attachments import run
        state = {
            "has_attachments": True,
            "attachments": [_jpeg_bytes()],
            "attachment_names": ["suspect_photo.jpg"],
        }
        result = run(state)
        assert result["attachment_types"] == ["person"]
        assert result["has_person"] is True

    def test_pdf_by_extension(self):
        from langraph_app.nodes.classify_attachments import run
        state = {
            "has_attachments": True,
            "attachments": [b"%PDF-1.4" + b"\x00" * 100],
            "attachment_names": ["report.pdf"],
        }
        result = run(state)
        assert result["attachment_types"] == ["document"]

    def test_vllm_called_for_ambiguous_filename(self):
        from langraph_app.nodes.classify_attachments import run
        llm_mock = _mock_llm_response("vehicle")
        state = {
            "has_attachments": True,
            "attachments": [_jpeg_bytes()],
            "attachment_names": ["img001.jpg"],
        }
        with (
            mock.patch("langraph_app.nodes.classify_attachments.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.classify_attachments.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
        ):
            result = run(state)
        assert result["attachment_types"] == ["vehicle"]

    def test_vllm_failure_returns_unknown(self):
        from langraph_app.nodes.classify_attachments import run
        state = {
            "has_attachments": True,
            "attachments": [_jpeg_bytes()],
            "attachment_names": ["img001.jpg"],
        }
        with (
            mock.patch("langraph_app.nodes.classify_attachments.get_vllm_client",
                       side_effect=RuntimeError("no backend")),
            mock.patch("langraph_app.nodes.classify_attachments.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
        ):
            result = run(state)
        assert result["attachment_types"] == ["unknown"]

    def test_hard_block_returns_empty(self):
        from langraph_app.nodes.classify_attachments import run
        result = run({"guardrail_hard_block": True, "has_attachments": True,
                      "attachments": [_jpeg_bytes()], "attachment_names": ["x.jpg"]})
        assert result == {}

    def test_multiple_attachments_mixed_types(self):
        from langraph_app.nodes.classify_attachments import run
        state = {
            "has_attachments": True,
            "attachments": [_jpeg_bytes(), _jpeg_bytes()],
            "attachment_names": ["screenshot_msg.png", "suspect_car.jpg"],
        }
        result = run(state)
        assert result["attachment_types"] == ["screenshot", "vehicle"]
        assert result["has_screenshot"] is True
        assert result["has_vehicle"] is True


# ── screenshot_node ───────────────────────────────────────────────────────────

class TestScreenshotNode:
    def test_extracts_transcript(self):
        from langraph_app.nodes.screenshot_node import run
        transcript = [
            {"sender": "John", "message": "I will hurt you", "timestamp": "10:30", "platform": "WhatsApp"}
        ]
        llm_mock = _mock_llm_response(json.dumps(transcript))
        state = {
            "has_screenshot": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["screenshot"],
        }
        with (
            mock.patch("langraph_app.nodes.screenshot_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.screenshot_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
        ):
            result = run(state)
        assert len(result["chat_transcript"]) == 1
        assert result["chat_transcript"][0]["sender"] == "John"
        assert result["chat_transcript"][0]["platform"] == "WhatsApp"

    def test_fallback_to_pytesseract_on_vllm_failure(self):
        from langraph_app.nodes.screenshot_node import run
        state = {
            "has_screenshot": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["screenshot"],
        }
        with (
            mock.patch("langraph_app.nodes.screenshot_node.get_vllm_client",
                       side_effect=RuntimeError("no backend")),
            mock.patch("langraph_app.nodes.screenshot_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
            mock.patch("langraph_app.nodes.screenshot_node.ocr_image_bytes",
                       return_value="I will shoot you"),
        ):
            result = run(state)
        assert len(result["chat_transcript"]) == 1
        assert "shoot" in result["chat_transcript"][0]["message"]

    def test_skips_non_screenshot_attachments(self):
        from langraph_app.nodes.screenshot_node import run
        llm_mock = _mock_llm_response("[]")
        state = {
            "has_screenshot": True,
            "attachments": [_jpeg_bytes(), _jpeg_bytes()],
            "attachment_types": ["vehicle", "screenshot"],
        }
        with (
            mock.patch("langraph_app.nodes.screenshot_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.screenshot_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
        ):
            result = run(state)
        # Only one LLM call (for index 1 which is "screenshot")
        assert llm_mock.invoke.call_count == 1

    def test_hard_block_returns_empty(self):
        from langraph_app.nodes.screenshot_node import run
        result = run({"guardrail_hard_block": True, "has_screenshot": True})
        assert result == {}


# ── vehicle_node ──────────────────────────────────────────────────────────────

class TestVehicleNode:
    def test_extracts_plate_and_details(self):
        from langraph_app.nodes.vehicle_node import run
        payload = {"plate": "ABC1234", "make_model": "Ford F-150",
                   "color": "blue", "damage_description": "dent on rear bumper"}
        llm_mock = _mock_llm_response(json.dumps(payload))
        state = {
            "has_vehicle": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["vehicle"],
        }
        with (
            mock.patch("langraph_app.nodes.vehicle_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.vehicle_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
        ):
            result = run(state)
        assert result["vehicle_extractions"][0]["plate"] == "ABC1234"
        assert result["vehicle_extractions"][0]["make_model"] == "Ford F-150"

    def test_plate_capped_at_20_chars(self):
        from langraph_app.nodes.vehicle_node import run
        payload = {"plate": "X" * 50, "make_model": "", "color": "", "damage_description": ""}
        llm_mock = _mock_llm_response(json.dumps(payload))
        state = {
            "has_vehicle": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["vehicle"],
        }
        with (
            mock.patch("langraph_app.nodes.vehicle_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.vehicle_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
        ):
            result = run(state)
        assert len(result["vehicle_extractions"][0]["plate"]) == 20

    def test_hard_block_returns_empty(self):
        from langraph_app.nodes.vehicle_node import run
        result = run({"guardrail_hard_block": True, "has_vehicle": True})
        assert result == {}


# ── id_document_node ──────────────────────────────────────────────────────────

class TestIDDocumentNode:
    def test_extracts_identity_fields(self):
        from langraph_app.nodes.id_document_node import run
        payload = {
            "full_name": "Jane Doe", "date_of_birth": "01/15/1990",
            "address": "123 Main St", "id_number": "D1234567",
            "issuer_state": "California", "expiry_date": "01/2028",
            "document_type": "drivers_license",
        }
        llm_mock = _mock_llm_response(json.dumps(payload))
        state = {
            "has_id_document": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["id_document"],
        }
        with (
            mock.patch("langraph_app.nodes.id_document_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.id_document_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
        ):
            result = run(state)
        doc = result["id_document_extractions"][0]
        assert doc["full_name"] == "Jane Doe"
        assert doc["document_type"] == "drivers_license"

    def test_invalid_doc_type_normalized_to_unknown(self):
        from langraph_app.nodes.id_document_node import run
        payload = {"full_name": "John", "document_type": "some_weird_type",
                   "date_of_birth": "", "address": "", "id_number": "",
                   "issuer_state": "", "expiry_date": ""}
        llm_mock = _mock_llm_response(json.dumps(payload))
        state = {
            "has_id_document": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["id_document"],
        }
        with (
            mock.patch("langraph_app.nodes.id_document_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.id_document_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
        ):
            result = run(state)
        assert result["id_document_extractions"][0]["document_type"] == "unknown"

    def test_hard_block_returns_empty(self):
        from langraph_app.nodes.id_document_node import run
        result = run({"guardrail_hard_block": True, "has_id_document": True})
        assert result == {}


# ── person_node ───────────────────────────────────────────────────────────────

class TestPersonNode:
    def test_appearance_only_when_not_suspect(self):
        from langraph_app.nodes.person_node import run
        llm_mock = _mock_llm_response("Male, early 30s, dark hair, blue jacket.")
        state = {
            "has_person": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["person"],
            "text_input": "A witness reported seeing a person near the scene.",
            "form_metadata": {},
        }
        with (
            mock.patch("langraph_app.nodes.person_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.person_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
            mock.patch("langraph_app.nodes.person_node.get_settings") as mock_settings,
        ):
            mock_settings.return_value.face_recognition_enabled = False
            mock_settings.return_value.known_offender_db_path = ""
            mock_settings.return_value.face_match_threshold = 0.60
            result = run(state)

        desc = result["person_descriptions"][0]
        assert desc["is_suspect"] is False
        assert desc["face_match_result"] == {}
        assert "30s" in desc["appearance"]

    def test_suspect_detected_from_text(self):
        from langraph_app.nodes.person_node import run
        llm_mock = _mock_llm_response("Female, mid 20s, red hoodie.")
        state = {
            "has_person": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["person"],
            "text_input": "The suspect fled the scene.",
            "form_metadata": {},
        }
        with (
            mock.patch("langraph_app.nodes.person_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.person_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
            mock.patch("langraph_app.nodes.person_node.get_settings") as mock_settings,
        ):
            mock_settings.return_value.face_recognition_enabled = False
            mock_settings.return_value.known_offender_db_path = ""
            mock_settings.return_value.face_match_threshold = 0.60
            result = run(state)

        assert result["person_descriptions"][0]["is_suspect"] is True
        assert result["person_descriptions"][0]["face_match_result"] == {}

    def test_suspect_photo_flag_in_form_metadata(self):
        from langraph_app.nodes.person_node import run
        llm_mock = _mock_llm_response("Male, tall, bald.")
        state = {
            "has_person": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["person"],
            "text_input": "No mention of suspect.",
            "form_metadata": {"suspect_photo": True},
        }
        with (
            mock.patch("langraph_app.nodes.person_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.person_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
            mock.patch("langraph_app.nodes.person_node.get_settings") as mock_settings,
        ):
            mock_settings.return_value.face_recognition_enabled = False
            mock_settings.return_value.known_offender_db_path = ""
            mock_settings.return_value.face_match_threshold = 0.60
            result = run(state)

        assert result["person_descriptions"][0]["is_suspect"] is True

    def test_face_recognition_not_called_when_disabled(self):
        from langraph_app.nodes.person_node import run
        llm_mock = _mock_llm_response("Person visible.")
        state = {
            "has_person": True,
            "attachments": [_jpeg_bytes()],
            "attachment_types": ["person"],
            "text_input": "The suspect ran away.",
            "form_metadata": {},
        }
        with (
            mock.patch("langraph_app.nodes.person_node.get_vllm_client",
                       return_value=llm_mock),
            mock.patch("langraph_app.nodes.person_node.to_data_url",
                       return_value="data:image/jpeg;base64,abc"),
            mock.patch("langraph_app.nodes.person_node.get_settings") as mock_settings,
            mock.patch("langraph_app.nodes.person_node._run_face_recognition") as mock_face,
        ):
            mock_settings.return_value.face_recognition_enabled = False
            mock_settings.return_value.known_offender_db_path = ""
            mock_settings.return_value.face_match_threshold = 0.60
            run(state)

        mock_face.assert_not_called()

    def test_hard_block_returns_empty(self):
        from langraph_app.nodes.person_node import run
        result = run({"guardrail_hard_block": True, "has_person": True})
        assert result == {}


# ── extract_node structured field injection ───────────────────────────────────

class TestExtractNodeStructuredFields:
    def test_chat_transcript_injected_as_text_context(self):
        """extract_node must include chat transcript text in the LLM message content."""
        from langraph_app.nodes.extract import _build_content
        state = {
            "text_input": "Report text",
            "chat_transcript": [
                {"sender": "Bob", "message": "I have a gun",
                 "timestamp": "22:00", "platform": "WhatsApp"}
            ],
            "vehicle_extractions": [],
            "id_document_extractions": [],
            "person_descriptions": [],
            "has_attachments": False,
            "image_data_urls": [],
        }
        content = _build_content(state, attachment_types=[])
        full_text = " ".join(
            c["text"] for c in content if c.get("type") == "text"
        )
        assert "I have a gun" in full_text
        assert "WhatsApp" in full_text

    def test_best_plate_wins_in_vehicle_context(self):
        from langraph_app.nodes.extract import _build_content
        state = {
            "text_input": "Report",
            "vehicle_extractions": [
                {"plate": "AB", "make_model": "Honda", "color": "red", "damage_description": ""},
                {"plate": "XYZ9876", "make_model": "Toyota", "color": "blue", "damage_description": ""},
            ],
            "chat_transcript": [],
            "id_document_extractions": [],
            "person_descriptions": [],
            "has_attachments": False,
            "image_data_urls": [],
        }
        content = _build_content(state, attachment_types=[])
        vehicle_blocks = [c["text"] for c in content if "Vehicle extraction" in c.get("text", "")]
        assert len(vehicle_blocks) == 1
        assert "XYZ9876" in vehicle_blocks[0]

    def test_structured_fields_overwrite_llm_output(self):
        from langraph_app.nodes.extract import _merge_structured_fields
        result_dict = _mock_extraction_result()
        state = {
            "chat_transcript": [{"sender": "X", "message": "test", "timestamp": "", "platform": "SMS"}],
            "id_document_extractions": [{"full_name": "Jane Doe", "document_type": "passport",
                                          "date_of_birth": "", "address": "", "id_number": "",
                                          "issuer_state": "", "expiry_date": ""}],
            "person_descriptions": [{"appearance": "tall", "is_suspect": True, "face_match_result": {}}],
        }
        merged = _merge_structured_fields(result_dict, state, attachment_types=["id_document"])
        assert merged["chat_transcript"][0]["sender"] == "X"
        assert merged["id_document"]["full_name"] == "Jane Doe"
        assert merged["person_descriptions"][0]["is_suspect"] is True
        assert merged["attachment_types"] == ["id_document"]


# ── Full intake graph smoke test ──────────────────────────────────────────────

class TestIntakeGraphAllTypes:
    def test_all_four_attachment_types_in_one_submission(self):
        """All 4 specialized nodes run and their outputs appear in validated_report."""
        import importlib
        import sys

        transcript = [{"sender": "Eve", "message": "threat msg",
                       "timestamp": "09:00", "platform": "SMS"}]
        vehicle_data = {"plate": "TEST123", "make_model": "Chevy Malibu",
                        "color": "black", "damage_description": ""}
        id_data = {"full_name": "John Smith", "date_of_birth": "01/01/1980",
                   "address": "456 Oak Ave", "id_number": "S9876543",
                   "issuer_state": "Texas", "expiry_date": "05/2027",
                   "document_type": "drivers_license"}

        extract_mock = mock.MagicMock()
        extract_mock.invoke.return_value = mock.MagicMock(
            model_dump=mock.MagicMock(return_value=_mock_extraction_result())
        )

        with (
            # Guardrails: pass through
            mock.patch("langraph_app.nodes.guardrails.run",
                       return_value={"guardrail_hard_block": False, "guardrail_flags": [],
                                     "guardrail_sanitized_text": "suspect in the area",
                                     "guardrail_sanitized_ocr": [], "audit_trail": []}),
            # Classification: all four types
            mock.patch("langraph_app.nodes.classify_attachments.run",
                       return_value={
                           "attachment_types": ["screenshot", "vehicle", "id_document", "person"],
                           "has_screenshot": True, "has_vehicle": True,
                           "has_id_document": True, "has_person": True,
                       }),
            # Specialized nodes
            mock.patch("langraph_app.nodes.screenshot_node.run",
                       return_value={"chat_transcript": transcript}),
            mock.patch("langraph_app.nodes.vehicle_node.run",
                       return_value={"vehicle_extractions": [vehicle_data]}),
            mock.patch("langraph_app.nodes.id_document_node.run",
                       return_value={"id_document_extractions": [id_data]}),
            mock.patch("langraph_app.nodes.person_node.run",
                       return_value={"person_descriptions": [
                           {"appearance": "tall male", "is_suspect": True, "face_match_result": {}}
                       ]}),
            # Base attachment nodes
            mock.patch("langraph_app.nodes.ocr.run",
                       return_value={"ocr_texts": [], "image_data_urls": []}),
            mock.patch("langraph_app.nodes.vision.run",
                       return_value={"visual_description": ""}),
            mock.patch("langraph_app.nodes.image_metadata.run",
                       return_value={"image_metadata": [], "image_metadata_conflicts": []}),
            # Extract + validate
            mock.patch("langraph_app.nodes.extract._get_llm", return_value=extract_mock),
            mock.patch("langraph_app.nodes.validate.hash_soc", return_value="hash123"),
            mock.patch("langraph_app.nodes.validate.quick_quotes", return_value=[]),
        ):
            # Reload the graph module so build() captures the patched .run functions
            sys.modules.pop("langraph_app.graphs.intake_graph", None)
            import langraph_app.graphs.intake_graph as _ig
            intake_graph = _ig.intake_graph

            state = intake_graph.invoke({
                "text_input": "suspect was seen near the school",
                "attachments": [b"\xff\xd8"] * 4,
                "attachment_names": [
                    "chat_screenshot.jpg", "suspect_car.jpg",
                    "drivers_license.jpg", "suspect_photo.jpg"
                ],
                "form_metadata": {"suspect_photo": True},
            })

        validated = state.get("validated_report", {})
        assert validated.get("chat_transcript") == transcript
        assert validated.get("vehicle_extractions") == [vehicle_data]
        assert validated.get("id_document_extractions") == [id_data]
        assert validated.get("person_descriptions")[0]["is_suspect"] is True
        assert len(validated.get("attachment_types", [])) == 4
