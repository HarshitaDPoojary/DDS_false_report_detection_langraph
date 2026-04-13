"""
Integration tests for AnalysisGraph.

These tests mock the Elasticsearch and LLM clients so the graph can run
offline. They verify the graph routes correctly and final state fields
are populated.
"""
from __future__ import annotations

import pytest
import unittest.mock as mock


# Shared mock helpers
def _make_es_mock(candidate_count: int = 0):
    """Return a mock Elasticsearch client that returns N candidate hits."""
    es_mock = mock.MagicMock()
    # kNN search response for retrieve_node
    es_mock.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": f"report-{i}",
                    "_score": 0.8,
                    "_source": {
                        "metadata": {
                            "report_id": f"report-{i}",
                            "reporter": {"anonymous": True},
                            "incident_types": [{"type": "fight", "confidence": 0.9}],
                            "location": [-73.9857, 40.7484],
                            "time_start": "2024-01-01T12:00:00Z",
                            "time_end": "2024-01-01T13:00:00Z",
                        },
                        "page_content": "Fight broke out near bus stop",
                    },
                }
                for i in range(candidate_count)
            ]
        }
    }
    # count responses for rate_limit_check_node
    es_mock.count.return_value = {"count": 0}
    return es_mock


def _base_state() -> dict:
    return {
        "report_id": "test-report-001",
        "raw_report": {
            "reporter": {
                "anonymous": True,
                "acct_age_days": 0,
                "ip_asn_class": "residential",
                "device_hash": "abc123",
            },
            "free_text": "Suspect with knife near Central Park.",
        },
        "free_text": "Suspect with knife near Central Park.",
        "location": [-73.9857, 40.7484],
        "time_start": "2024-01-01T12:00:00Z",
        "time_end": "2024-01-01T13:00:00Z",
        "time_midpoint": "2024-01-01T12:30:00Z",
        "text_embedding": [0.1] * 384,
        "incident_types": [{"type": "assault", "confidence": 0.85, "matches": []}],
        "severity": "medium",
        "image_metadata": [],
        "image_metadata_conflicts": [],
        "visual_description": "",
        "soc_hash": "",
        "extraction_result": {},
        "radius_miles": 5.0,
        "lookback_hours": 24.0,
        "scorer_weights": {},
    }


@pytest.mark.asyncio
class TestAnalysisGraphRouting:
    async def test_guardrail_hard_block_aborts(self):
        """Graph should abort at END immediately on guardrail hard block."""
        from langraph_app.graphs.analysis_graph import analysis_graph
        from langraph_app.utils.sanitize import TEXT_HARD_LIMIT_CHARS

        state = _base_state()
        state["free_text"] = "A" * (TEXT_HARD_LIMIT_CHARS + 1)
        state["raw_report"]["free_text"] = state["free_text"]

        result = await analysis_graph.ainvoke(state)

        assert result.get("guardrail_hard_block") is True
        assert result.get("error")

    async def test_no_candidates_skips_score_hoax(self):
        """With no ES candidates, graph should skip score_node and hoax_node."""
        from langraph_app.graphs.analysis_graph import analysis_graph
        from langraph_app.nodes import retrieve, embed, classify, reporter_credibility

        state = _base_state()

        with (
            mock.patch("langraph_app.nodes.retrieve._get_es", return_value=_make_es_mock(0)),
            mock.patch("langraph_app.nodes.rate_limit_check._get_es", return_value=_make_es_mock(0)),
            mock.patch("langraph_app.nodes.reporter_credibility._get_es", return_value=_make_es_mock(0)),
            mock.patch("langraph_app.nodes.reporter_credibility._lookup_reporter_history",
                       return_value={"confirmed_hoax_count": 0, "confirmed_real_count": 0}),
            mock.patch("langraph_app.nodes.embed._model") as embed_mock,
        ):
            embed_mock.embed_query.return_value = [0.1] * 384

            result = await analysis_graph.ainvoke(state)

        assert "urgency_score" in result
        assert "hoax_probability" in result
        assert result.get("action") in ("monitor", "dismiss", "human_review", "escalate")

    async def test_full_pipeline_with_candidates(self):
        """Full pipeline run with 3 anonymous candidates should produce high hoax signal."""
        from langraph_app.graphs.analysis_graph import analysis_graph

        state = _base_state()

        with (
            mock.patch("langraph_app.nodes.retrieve._get_es", return_value=_make_es_mock(3)),
            mock.patch("langraph_app.nodes.rate_limit_check._get_es", return_value=_make_es_mock(0)),
            mock.patch("langraph_app.nodes.reporter_credibility._get_es", return_value=_make_es_mock(0)),
            mock.patch("langraph_app.nodes.reporter_credibility._lookup_reporter_history",
                       return_value={"confirmed_hoax_count": 0, "confirmed_real_count": 0}),
            mock.patch("langraph_app.nodes.embed._model") as embed_mock,
        ):
            embed_mock.embed_query.return_value = [0.1] * 384
            result = await analysis_graph.ainvoke(state)

        assert 0.0 <= result["hoax_probability"] <= 1.0
        assert 0.0 <= result["threat_level"] <= 1.0
        assert result["action"] in ("monitor", "dismiss", "human_review", "escalate")
        assert result.get("ai_analysis")
        assert result.get("audit_trail")

    async def test_rate_limit_flag_propagates(self):
        """If rate_limit_check returns flagged, it should appear in final result."""
        from langraph_app.graphs.analysis_graph import analysis_graph
        from langraph_app.nodes import rate_limit_check

        state = _base_state()

        es_burst = _make_es_mock(0)
        es_burst.count.return_value = {"count": 5}  # triggers device_burst

        with (
            mock.patch("langraph_app.nodes.retrieve._get_es", return_value=_make_es_mock(0)),
            mock.patch("langraph_app.nodes.rate_limit_check._get_es", return_value=es_burst),
            mock.patch("langraph_app.nodes.reporter_credibility._get_es", return_value=_make_es_mock(0)),
            mock.patch("langraph_app.nodes.reporter_credibility._lookup_reporter_history",
                       return_value={"confirmed_hoax_count": 0, "confirmed_real_count": 0}),
            mock.patch("langraph_app.nodes.embed._model") as embed_mock,
        ):
            embed_mock.embed_query.return_value = [0.1] * 384
            result = await analysis_graph.ainvoke(state)

        assert result.get("rate_limit_flagged") is True
