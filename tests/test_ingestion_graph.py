"""
Integration tests for IngestionGraph.

Tests:
  - Successful embed + classify + index run
  - ES index failure returns error in state
  - Embedding produced with correct dimensions
"""
from __future__ import annotations

import pytest
import unittest.mock as mock


def _base_state() -> dict:
    return {
        "report_id": "ingest-001",
        "raw_report": {
            "report_id": "ingest-001",
            "free_text": "Fight near bus stop.",
        },
        "free_text": "Fight near bus stop.",
        "location": [-73.9857, 40.7484],
        "time_start": "2024-01-01T12:00:00Z",
        "time_end": "2024-01-01T13:00:00Z",
        "time_midpoint": "2024-01-01T12:30:00Z",
        "text_embedding": [],
        "incident_types": [],
        "severity": "low",
        "validated_report": {"report_id": "ingest-001"},
        "image_metadata": [],
        "visual_description": "",
    }


@pytest.mark.asyncio
class TestIngestionGraph:
    async def test_successful_run(self):
        from langraph_app.graphs.ingestion_graph import ingestion_graph

        es_store_mock = mock.MagicMock()
        es_store_mock.add_documents.return_value = None

        with (
            mock.patch("langraph_app.nodes.embed._model") as embed_mock,
            mock.patch("langraph_app.nodes.classify._get_regex_types",
                       return_value=[{"type": "fight", "confidence": 0.9, "matches": []}]),
            mock.patch("langraph_app.nodes.index._get_store", return_value=es_store_mock),
        ):
            embed_mock.embed_query.return_value = [0.1] * 384
            result = await ingestion_graph.ainvoke(_base_state())

        assert result.get("indexed") is True
        assert not result.get("error")

    async def test_embedding_dimension(self):
        from langraph_app.graphs.ingestion_graph import ingestion_graph

        es_store_mock = mock.MagicMock()
        es_store_mock.add_documents.return_value = None

        with (
            mock.patch("langraph_app.nodes.embed._model") as embed_mock,
            mock.patch("langraph_app.nodes.classify._get_regex_types",
                       return_value=[{"type": "assault", "confidence": 0.85, "matches": []}]),
            mock.patch("langraph_app.nodes.index._get_store", return_value=es_store_mock),
        ):
            embed_mock.embed_query.return_value = [0.5] * 384
            result = await ingestion_graph.ainvoke(_base_state())

        assert len(result.get("text_embedding", [])) == 384

    async def test_es_error_returns_error_in_state(self):
        from langraph_app.graphs.ingestion_graph import ingestion_graph

        es_store_mock = mock.MagicMock()
        es_store_mock.add_documents.side_effect = ConnectionError("ES unreachable")

        with (
            mock.patch("langraph_app.nodes.embed._model") as embed_mock,
            mock.patch("langraph_app.nodes.classify._get_regex_types",
                       return_value=[{"type": "fight", "confidence": 0.9, "matches": []}]),
            mock.patch("langraph_app.nodes.index._get_store", return_value=es_store_mock),
        ):
            embed_mock.embed_query.return_value = [0.1] * 384
            result = await ingestion_graph.ainvoke(_base_state())

        assert result.get("error")
        assert result.get("indexed") is False
