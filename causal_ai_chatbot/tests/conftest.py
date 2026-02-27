"""
Shared fixtures for the Causal AI Chatbot test suite.

These fixtures provide reusable test data, mock objects, and session state
that individual test modules can use.
"""

import os
import sys
import uuid
import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# ── Add parent directory to path so we can import chatbot_server ──
CHATBOT_DIR = Path(__file__).parent.parent.resolve()
if str(CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_DIR))

# Ensure local LLM mode in tests unless explicitly overridden.
os.environ.setdefault("LLM_PROVIDER", "ollama")


# ── Import after path & env setup ──
import chatbot_server
import app.chatbot_server as app_chatbot_server
from chatbot_server import (
    ToolRegistry,
    CausalTools,
    AgentOrchestrator,
    SessionState,
    ChatMessage,
    sessions,
    app,
)


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_csv_path():
    """Path to the sample CSV file in data/."""
    path = CHATBOT_DIR / "data" / "sampled_data_1000.csv"
    assert path.exists(), f"Test data not found at {path}"
    return str(path)


@pytest.fixture
def sample_dataframe(sample_csv_path):
    """Load the sample CSV into a DataFrame."""
    return pd.read_csv(sample_csv_path)


@pytest.fixture
def sample_variables():
    """Variable names from the sample dataset."""
    return ["x1", "x2", "x3"]


# ============================================================================
# DAG Fixtures
# ============================================================================

@pytest.fixture
def sample_dag(sample_variables):
    """A simple 3-variable DAG: x1 → x2 → x3."""
    vars = sample_variables
    n = len(vars)
    adj = np.zeros((n, n), dtype=int)
    adj[0, 1] = 1  # x1 → x2
    adj[1, 2] = 1  # x2 → x3
    return {
        "adjacency_matrix": adj.tolist(),
        "variables": vars,
        "edges": [("x1", "x2"), ("x2", "x3")],
        "llm_explanation": "Test DAG: x1 causes x2, x2 causes x3",
    }


@pytest.fixture
def sample_dag_full(sample_variables):
    """A fully connected 3-variable DAG: x1 → x2, x1 → x3, x2 → x3."""
    vars = sample_variables
    n = len(vars)
    adj = np.zeros((n, n), dtype=int)
    adj[0, 1] = 1  # x1 → x2
    adj[0, 2] = 1  # x1 → x3
    adj[1, 2] = 1  # x2 → x3
    return {
        "adjacency_matrix": adj.tolist(),
        "variables": vars,
        "edges": [("x1", "x2"), ("x1", "x3"), ("x2", "x3")],
        "llm_explanation": "Test DAG: fully connected",
    }


# ============================================================================
# Session Fixtures
# ============================================================================

@pytest.fixture
def session_id():
    """A fresh unique session ID for each test."""
    return str(uuid.uuid4())


@pytest.fixture
def clean_sessions():
    """Clear the global sessions dict before and after test."""
    sessions.clear()
    yield sessions
    sessions.clear()


@pytest.fixture
def session_with_data(session_id, sample_csv_path, sample_dataframe, clean_sessions):
    """A session that already has data uploaded."""
    sessions[session_id] = {
        "session_id": session_id,
        "messages": [],
        "created_at": "2025-01-01T00:00:00",
        "current_step": "initial",
        "data_path": sample_csv_path,
        "data_df": sample_dataframe.to_dict("records"),
        "data_info": {
            "shape": list(sample_dataframe.shape),
            "columns": list(sample_dataframe.columns),
            "dtypes": {col: str(dt) for col, dt in sample_dataframe.dtypes.items()},
        },
    }
    return session_id


@pytest.fixture
def session_with_dag(session_with_data, sample_dag):
    """A session that has data + a proposed DAG."""
    sid = session_with_data
    sessions[sid]["proposed_dag"] = sample_dag
    sessions[sid]["current_step"] = "dag_proposed"
    return sid


# ============================================================================
# Orchestrator Fixtures
# ============================================================================

@pytest.fixture
def orchestrator(session_with_data):
    """An AgentOrchestrator initialised with a data-loaded session."""
    return AgentOrchestrator(session_with_data)


@pytest.fixture
def orchestrator_with_dag(session_with_dag, sample_dag):
    """An AgentOrchestrator with data + DAG already proposed."""
    orch = AgentOrchestrator(session_with_dag)
    orch.state.proposed_dag = sample_dag
    orch.state.current_step = "dag_proposed"
    return orch


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_llm():
    """Mock the global llm_client so no real LLM calls are made."""
    mock_client = MagicMock()

    # Default: LLM returns a simple text response (no tool call)
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a mock LLM response."
    mock_choice.message.tool_calls = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    with patch.object(chatbot_server, "llm_client", mock_client), patch.object(app_chatbot_server, "llm_client", mock_client):
        yield mock_client


@pytest.fixture
def mock_llm_with_tool_call():
    """Mock the llm_client to return a specific tool call."""
    mock_client = MagicMock()

    def _make_tool_response(tool_name: str, tool_args: dict):
        """Helper: configure the mock to return a specific tool call."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = tool_name
        mock_tool_call.function.arguments = json.dumps(tool_args)

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

    mock_client.make_tool_response = _make_tool_response

    with patch.object(chatbot_server, "llm_client", mock_client), patch.object(app_chatbot_server, "llm_client", mock_client):
        yield mock_client


# ============================================================================
# FastAPI Test Client
# ============================================================================

@pytest.fixture
def client():
    """FastAPI TestClient for HTTP endpoint testing."""
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_mcp_for_tests():
    """
    Prevent tests from spawning real MCP wrapper subprocesses.

    Individual tests that need MCP behavior can still patch `_call_mcp`
    explicitly with their own return payloads.
    """
    def _fake_call_mcp(tool_name: str, payload: dict):
        vars_list = list(payload.get("vars", []))
        if tool_name == "propose_dag":
            edges = [[vars_list[i], vars_list[i + 1]] for i in range(max(0, len(vars_list) - 1))]
            n = len(vars_list)
            adj = [[0 for _ in range(n)] for _ in range(n)]
            idx = {v: i for i, v in enumerate(vars_list)}
            for p, c in edges:
                adj[idx[p]][idx[c]] = 1
            return {
                "success": True,
                "data": {
                    "variables": vars_list,
                    "edges": [{"from": p, "to": c} for p, c in edges],
                    "adjacency_matrix": adj,
                    "llm_explanation": "Mock MCP DAG",
                },
                "artifacts": [],
                "tables": {},
                "messages": [],
                "error": None,
            }
        if tool_name == "test_dag":
            return {
                "success": True,
                "data": {"consistent": True, "rejected_count": 0, "tests": []},
                "artifacts": [],
                "tables": {},
                "messages": [],
                "error": None,
            }
        if tool_name == "fit_model":
            return {
                "success": True,
                "data": {
                    "experiment_dir": payload.get("experiment_dir", "/tmp/test-exp"),
                    "config_path": "/tmp/test-exp/configuration.json",
                    "loss_history": {},
                    "artifact_manifest": [],
                },
                "artifacts": [],
                "tables": {},
                "messages": [],
                "error": None,
            }
        if tool_name == "sample":
            return {
                "success": True,
                "data": {"samples": {}, "artifact_manifest": []},
                "artifacts": [],
                "tables": {},
                "messages": [],
                "error": None,
            }
        if tool_name == "compute_ate":
            return {
                "success": True,
                "data": {
                    "ate": 0.0,
                    "X": payload.get("X", "x1"),
                    "Y": payload.get("Y", "x2"),
                    "x_treated": payload.get("x_treated", 1.0),
                    "x_control": payload.get("x_control", 0.0),
                    "y_treated_mean": 0.0,
                    "y_treated_std": 0.0,
                    "y_control_mean": 0.0,
                    "y_control_std": 0.0,
                    "artifact_manifest": [],
                },
                "artifacts": [],
                "tables": {},
                "messages": [],
                "error": None,
            }
        return {
            "success": False,
            "data": {},
            "artifacts": [],
            "tables": {},
            "messages": [],
            "error": f"Unsupported mock MCP tool: {tool_name}",
        }

    with patch.object(
        CausalTools,
        "_call_mcp",
        side_effect=_fake_call_mcp,
    ):
        yield
