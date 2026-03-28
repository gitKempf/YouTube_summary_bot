"""Tests for FastAPI endpoints."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from webapp.app import create_app


@pytest.fixture
def client():
    app = create_app(testing=True)
    return TestClient(app)


class TestVideosEndpoint:
    def test_returns_video_list(self, client):
        with patch("webapp.routes.videos.get_qdrant") as mock_qdrant:
            mock_qdrant.return_value.scroll.return_value = ([
                MagicMock(payload={"video_id": "abc123", "user_id": "tg_42",
                                   "language_code": "en", "stored_at": "2026-03-28"}),
            ], None)
            resp = client.get("/api/videos", params={"user_id": "tg_42"})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["video_id"] == "abc123"


class TestSettingsEndpoint:
    def test_get_settings(self, client):
        with patch("webapp.routes.settings.get_qdrant") as mock_qdrant:
            mock_qdrant.return_value.count.return_value = MagicMock(count=5)
            resp = client.get("/api/settings", params={"user_id": "tg_42"})

        assert resp.status_code == 200
        data = resp.json()
        assert "user_id" in data
        assert "claim_count" in data


class TestExportEndpoint:
    def test_export_returns_zip(self, client):
        with patch("webapp.routes.export.build_vault_zip") as mock_build:
            mock_build.return_value = b"PK\\x03\\x04fake_zip_data"
            resp = client.get("/api/export/obsidian", params={"user_id": "tg_42"})

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
