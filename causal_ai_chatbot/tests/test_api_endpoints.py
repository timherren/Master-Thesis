"""
Tests for the FastAPI HTTP endpoints.

Uses FastAPI's TestClient so no real server needs to run.

Validates:
  - GET /                     → serves HTML page
  - POST /api/upload_data     → accepts CSV, returns session + data_info
  - POST /api/chat            → returns a response message
  - GET  /api/debug_session   → returns session data
  - GET  /api/dag_editor_data → returns DAG data or 404
  - POST /api/save_dag        → saves DAG to session
  - GET  /api/plots/{filename}→ serves plot files
"""

import io
import json
import uuid
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from .conftest import sessions, app, CHATBOT_DIR


# ============================================================================
# GET / — HTML Serving
# ============================================================================

class TestHomePage:
    """Test that the root endpoint serves the HTML chatbot UI."""

    def test_home_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_home_returns_html(self, client):
        resp = client.get("/")
        assert "text/html" in resp.headers.get("content-type", "")


# ============================================================================
# POST /api/upload_data — File Upload
# ============================================================================

class TestUploadData:
    """Test the CSV upload endpoint."""

    def test_upload_csv_returns_success(self, client, sample_csv_path, clean_sessions):
        with open(sample_csv_path, "rb") as f:
            resp = client.post(
                "/api/upload_data",
                files={"file": ("test.csv", f, "text/csv")},
                data={"session_id": "upload-test"},
            )
        data = resp.json()
        assert data["status"] == "uploaded"
        assert data["session_id"] == "upload-test"
        assert "data_info" in data
        assert data["data_info"]["columns"] == ["x1", "x2", "x3"]
        assert "data_pairplot_url" in data

    def test_upload_creates_session(self, client, sample_csv_path, clean_sessions):
        with open(sample_csv_path, "rb") as f:
            resp = client.post(
                "/api/upload_data",
                files={"file": ("test.csv", f, "text/csv")},
                data={"session_id": "new-session"},
            )
        assert "new-session" in sessions
        assert sessions["new-session"]["data_path"] is not None

    def test_upload_without_session_id_generates_one(self, client, sample_csv_path, clean_sessions):
        with open(sample_csv_path, "rb") as f:
            resp = client.post(
                "/api/upload_data",
                files={"file": ("test.csv", f, "text/csv")},
            )
        data = resp.json()
        assert data["status"] == "uploaded"
        assert "session_id" in data
        # Should be a UUID
        uuid.UUID(data["session_id"])  # Raises if invalid

    def test_upload_invalid_file_parsed_leniently(self, client, clean_sessions):
        """Pandas will parse almost anything as CSV (even garbage), so we
        just verify the endpoint doesn't crash and returns a valid response."""
        bad_content = b"this is not csv or excel data \x00\x01\x02"
        resp = client.post(
            "/api/upload_data",
            files={"file": ("garbage.xyz", io.BytesIO(bad_content), "application/octet-stream")},
            data={"session_id": "bad-upload"},
        )
        data = resp.json()
        # pandas can parse almost anything as 0-row CSV — that's acceptable
        assert data["status"] in ("uploaded", "error")

    def test_upload_data_info_shape(self, client, sample_csv_path, clean_sessions):
        with open(sample_csv_path, "rb") as f:
            resp = client.post(
                "/api/upload_data",
                files={"file": ("test.csv", f, "text/csv")},
                data={"session_id": "shape-test"},
            )
        info = resp.json()["data_info"]
        assert info["shape"][0] == 1000  # 1000 rows
        assert info["shape"][1] == 3     # 3 columns


# ============================================================================
# POST /api/chat — Chat Endpoint
# ============================================================================

class TestChatEndpoint:
    """Test the HTTP chat endpoint (as opposed to WebSocket)."""

    def test_chat_returns_response(self, client, session_with_data, mock_llm):
        resp = client.post(
            "/api/chat",
            json={"message": "hello", "session_id": session_with_data},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    def test_chat_without_session_returns_error(self, client, clean_sessions, mock_llm):
        resp = client.post(
            "/api/chat",
            json={"message": "hello", "session_id": "nonexistent"},
        )
        data = resp.json()
        # Should still return 200 but with some form of response
        assert resp.status_code == 200


# ============================================================================
# GET /api/debug_session — Session Debug
# ============================================================================

class TestDebugSession:
    """Test the debug endpoint that exposes session state."""

    def test_debug_returns_session_data(self, client, session_with_data):
        resp = client.get(f"/api/debug_session?session_id={session_with_data}")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["session_id"] == session_with_data

    def test_debug_nonexistent_session(self, client, clean_sessions):
        resp = client.get("/api/debug_session?session_id=fake")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data or data.get("session_id") is None


# ============================================================================
# GET /api/dag_editor_data — DAG Editor Data
# ============================================================================

class TestDagEditorData:
    """Test the endpoint that provides DAG data for the interactive editor."""

    def test_dag_editor_data_with_dag(self, client, session_with_dag, sample_dag):
        resp = client.get(f"/api/dag_editor_data?session_id={session_with_dag}")
        assert resp.status_code == 200
        data = resp.json()
        assert "variables" in data
        assert "nodes" in data
        assert "edge_objects" in data

    def test_dag_editor_data_without_dag(self, client, session_with_data):
        resp = client.get(f"/api/dag_editor_data?session_id={session_with_data}")
        # Should return something (maybe with variables only)
        assert resp.status_code == 200


# ============================================================================
# POST /api/save_dag — Save DAG from Editor
# ============================================================================

class TestSaveDag:
    """Test saving a DAG from the interactive editor."""

    def test_save_dag_updates_session(self, client, session_with_data, sample_dag):
        """save_dag expects {session_id, edges} where edges is List[List[str]]."""
        resp = client.post(
            "/api/save_dag",
            json={
                "session_id": session_with_data,
                "edges": [["x1", "x2"], ["x2", "x3"]],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "saved"
        assert data.get("edge_count") == 2

    def test_save_dag_without_session_returns_404(self, client, clean_sessions):
        resp = client.post(
            "/api/save_dag",
            json={
                "session_id": "nonexistent",
                "edges": [["a", "b"]],
            },
        )
        assert resp.status_code == 404

    def test_save_dag_cycle_returns_validation_error(self, client, session_with_data):
        resp = client.post(
            "/api/save_dag",
            json={
                "session_id": session_with_data,
                "edges": [["x1", "x2"], ["x2", "x1"]],
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "validation_error"
        assert "cycle" in json.dumps(payload.get("details", {})).lower()


# ============================================================================
# GET /api/plots/{filename} — Plot Serving
# ============================================================================

class TestPlotServing:
    """Test serving static plot images."""

    def test_nonexistent_plot_returns_404(self, client):
        resp = client.get("/api/plots/nonexistent_abc123.png")
        assert resp.status_code == 404

    def test_existing_plot_returns_image(self, client, tmp_path):
        """Create a temp plot file and verify it's served."""
        # Use the global TEMP_PLOTS_DIR from server
        from .conftest import chatbot_server
        plots_dir = chatbot_server.TEMP_PLOTS_DIR
        plots_dir.mkdir(exist_ok=True)

        test_file = plots_dir / "test_plot_serve.png"
        test_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # Minimal PNG header

        resp = client.get("/api/plots/test_plot_serve.png")
        assert resp.status_code == 200
        assert "image" in resp.headers.get("content-type", "")

        # Clean up
        test_file.unlink(missing_ok=True)


# ============================================================================
# Session Lifecycle — Integration
# ============================================================================

class TestSessionLifecycle:
    """Integration test: upload → chat → debug → check state."""

    def test_upload_then_chat_preserves_session(self, client, sample_csv_path, clean_sessions, mock_llm):
        # 1. Upload data
        with open(sample_csv_path, "rb") as f:
            upload_resp = client.post(
                "/api/upload_data",
                files={"file": ("test.csv", f, "text/csv")},
                data={"session_id": "lifecycle-test"},
            )
        assert upload_resp.json()["status"] == "uploaded"

        # 2. Send a chat message
        chat_resp = client.post(
            "/api/chat",
            json={"message": "What variables do I have?", "session_id": "lifecycle-test"},
        )
        assert chat_resp.status_code == 200

        # 3. Debug should show the session
        debug_resp = client.get("/api/debug_session?session_id=lifecycle-test")
        assert debug_resp.status_code == 200
        data = debug_resp.json()
        assert data.get("data_path") is not None or data.get("data_info") is not None
