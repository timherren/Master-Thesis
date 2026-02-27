"""
Tests for AgentOrchestrator — the AI Agent that routes user messages to tools.

Validates:
  - Orchestrator initialisation and state restoration
  - State transitions after tool execution
  - The agent loop: LLM selects tool → tool executes → response formatted
  - _execute_tool dispatches correctly for each tool
  - _format_tool_results produces valid markdown
  - Fallback to workflow routing on agent failure
"""

import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from .conftest import (
    AgentOrchestrator,
    CausalTools,
    SessionState,
    sessions,
    chatbot_server,
)


# ============================================================================
# Initialization Tests
# ============================================================================

class TestOrchestratorInit:
    """Test that the orchestrator correctly initialises from session state."""

    def test_creates_with_initial_step(self, session_with_data):
        orch = AgentOrchestrator(session_with_data)
        assert orch.state.current_step == "initial"
        assert orch.session_id == session_with_data

    def test_restores_data_path_from_session(self, session_with_data, sample_csv_path):
        orch = AgentOrchestrator(session_with_data)
        assert orch.state.data_path == sample_csv_path

    def test_restores_dag_from_session(self, session_with_dag, sample_dag):
        orch = AgentOrchestrator(session_with_dag)
        assert orch.state.proposed_dag is not None
        assert orch.state.proposed_dag["variables"] == sample_dag["variables"]

    def test_restores_step_from_session(self, session_with_dag):
        orch = AgentOrchestrator(session_with_dag)
        assert orch.state.current_step == "dag_proposed"

    def test_fresh_session_has_initial_step(self, session_id, clean_sessions):
        sessions[session_id] = {
            "session_id": session_id,
            "messages": [],
            "current_step": "initial",
        }
        orch = AgentOrchestrator(session_id)
        assert orch.state.current_step == "initial"


# ============================================================================
# State Transition Tests
# ============================================================================

class TestStateTransitions:
    """Test that _execute_tool updates current_step correctly."""

    @pytest.mark.asyncio
    async def test_propose_dag_sets_dag_proposed(self, orchestrator, mock_llm):
        """After propose_dag, step should be 'dag_proposed'."""
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            "x1,x2\nx2,x3"
        )

        result = await orchestrator._execute_tool("propose_dag", {"vars": ["x1", "x2", "x3"]})

        assert result["success"]
        assert orchestrator.state.current_step == "dag_proposed"
        assert orchestrator.state.proposed_dag is not None

    @pytest.mark.asyncio
    async def test_test_dag_sets_dag_tested(self, orchestrator_with_dag, mock_llm):
        """After test_dag, step should be 'dag_tested'."""
        # Mock the R checker to return a simple result
        mock_result = {
            "consistent": True,
            "rejected_count": 0,
            "tests": [{"ci": "x1 _||_ x3 | x2", "rejected": False, "p_value": 0.5}],
        }
        with patch.object(CausalTools, "test_dag_consistency", return_value=mock_result):
            result = await orchestrator_with_dag._execute_tool("test_dag", {})

        assert result["success"]
        assert orchestrator_with_dag.state.current_step == "dag_tested"
        assert orchestrator_with_dag.state.ci_test_results is not None

    @pytest.mark.asyncio
    async def test_fit_model_sets_model_fitted(self, orchestrator_with_dag):
        """After fit_model, step should be 'model_fitted'."""
        mock_fit = {"experiment_dir": "/tmp/test", "loss_history": {}}
        with patch.object(CausalTools, "fit_tramdag_model", return_value=mock_fit):
            with patch.object(CausalTools, "_create_loss_plot", return_value=None):
                with patch.object(CausalTools, "_create_distribution_plot", return_value=None):
                    with patch.object(CausalTools, "_create_dag_plot", return_value=None):
                        result = await orchestrator_with_dag._execute_tool(
                            "fit_model",
                            {"epochs": 100, "learning_rate": 0.01, "batch_size": 512},
                        )

        assert result["success"]
        assert orchestrator_with_dag.state.current_step == "model_fitted"

    @pytest.mark.asyncio
    async def test_open_dag_editor_does_not_change_step(self, orchestrator_with_dag):
        """open_dag_editor should not change the workflow step."""
        original_step = orchestrator_with_dag.state.current_step
        result = await orchestrator_with_dag._execute_tool("open_dag_editor", {})

        assert result["success"]
        assert orchestrator_with_dag.state.current_step == original_step


# ============================================================================
# _execute_tool Error Handling
# ============================================================================

class TestExecuteToolErrors:
    """Test that _execute_tool handles missing state gracefully."""

    @pytest.mark.asyncio
    async def test_test_dag_without_dag_returns_error(self, orchestrator):
        """test_dag with no DAG should return an error."""
        orchestrator.state.proposed_dag = None
        result = await orchestrator._execute_tool("test_dag", {})
        assert not result["success"]
        assert result["error"] is not None
        assert "DAG" in result["error"]

    @pytest.mark.asyncio
    async def test_fit_model_without_dag_returns_error(self, orchestrator):
        """fit_model with no DAG should return an error."""
        orchestrator.state.proposed_dag = None
        result = await orchestrator._execute_tool("fit_model", {})
        assert not result["success"]
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_sample_without_model_returns_error(self, orchestrator):
        """sample with no fitted model should return an error."""
        orchestrator.state.experiment_dir = None
        result = await orchestrator._execute_tool("sample", {})
        assert not result["success"]
        assert "not fitted" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_compute_ate_without_model_returns_error(self, orchestrator):
        """compute_ate now starts guided intake (X/Y confirmation) before execution."""
        orchestrator.state.experiment_dir = None
        result = await orchestrator._execute_tool("compute_ate", {"X": "x1", "Y": "x3"})
        assert result["success"] is True
        assert "_guided_prompt" in result["data"]
        assert orchestrator.state.pending_tool == "compute_ate"

    @pytest.mark.asyncio
    async def test_compute_ate_without_variables_returns_error(self, orchestrator):
        """compute_ate without X/Y should prompt for guided variable selection."""
        orchestrator.state.experiment_dir = "/tmp/fake"
        result = await orchestrator._execute_tool("compute_ate", {})
        assert result["success"] is True
        assert "_guided_prompt" in result["data"]
        assert orchestrator.state.pending_tool == "compute_ate"

    @pytest.mark.asyncio
    async def test_show_associations_without_data_returns_error(self, orchestrator):
        """show_associations with no data loaded should error."""
        orchestrator.state.data_df = None
        result = await orchestrator._execute_tool("show_associations", {})
        assert not result["success"]
        assert "data" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, orchestrator):
        """Unknown tool name should return an error."""
        result = await orchestrator._execute_tool("nonexistent_tool", {})
        assert not result["success"]
        assert "Unknown tool" in result["error"]


# ============================================================================
# _execute_tool Success Paths
# ============================================================================

class TestExecuteToolSuccess:
    """Test successful tool execution (with mocks where needed)."""

    @pytest.mark.asyncio
    async def test_show_associations_returns_correlation(self, orchestrator, sample_dataframe):
        """show_associations should return a correlation matrix string."""
        orchestrator.state.data_df = sample_dataframe
        result = await orchestrator._execute_tool("show_associations", {})

        assert result["success"]
        assert "correlation_matrix" in result["data"]
        assert "x1" in result["data"]["correlation_matrix"]

    @pytest.mark.asyncio
    async def test_open_dag_editor_returns_action(self, orchestrator):
        result = await orchestrator._execute_tool("open_dag_editor", {})
        assert result["success"]
        assert result["data"]["action"] == "open_dag_editor"

    @pytest.mark.asyncio
    async def test_propose_dag_fills_vars_from_data(self, orchestrator, mock_llm):
        """If vars not provided, should use data columns.
        Note: x1/x2/x3 are generic names, so should trigger generic path."""
        result = await orchestrator._execute_tool("propose_dag", {})
        assert result["success"]
        # x1/x2/x3 are generic → empty edges + _generic_names marker
        assert result["data"]["_generic_names"] is True
        assert result["data"]["edges"] == []
        assert "x1" in result["data"]["variables"]

    @pytest.mark.asyncio
    async def test_fit_model_prefers_mcp_artifacts(self, orchestrator_with_dag):
        """When MCP artifacts exist, fit_model should use them instead of local plotting."""
        mock_fit = {
            "experiment_dir": "/tmp/test",
            "loss_history": {},
            "mcp_artifacts": ["/tmp/loss_history.png"],
            "artifact_manifest": [{"kind": "loss_history", "path": "/tmp/loss_history.png"}],
        }
        with patch.object(CausalTools, "fit_tramdag_model", return_value=mock_fit):
            with patch.object(
                AgentOrchestrator,
                "_import_and_store_mcp_artifacts",
                return_value=["/api/plots/test_loss.png"],
            ):
                with patch.object(CausalTools, "_create_loss_plot", side_effect=AssertionError("fallback should not run")):
                    result = await orchestrator_with_dag._execute_tool(
                        "fit_model",
                        {"epochs": 100, "learning_rate": 0.01, "batch_size": 512},
                    )

        assert result["success"]
        assert "/api/plots/test_loss.png" in result["plots"]

    @pytest.mark.asyncio
    async def test_create_plots_uses_stored_mcp_urls(self, orchestrator_with_dag):
        """create_plots should prefer already-stored MCP plot URLs."""
        orchestrator_with_dag.state.experiment_dir = "/tmp/exp"
        sessions[orchestrator_with_dag.session_id]["mcp_plot_urls_by_kind"] = {
            "loss_history": "/api/plots/existing_loss.png"
        }
        with patch.object(CausalTools, "_create_loss_plot", side_effect=AssertionError("fallback should not run")):
            result = await orchestrator_with_dag._execute_tool("create_plots", {"plot_types": ["loss"]})
        assert result["success"]
        assert "/api/plots/existing_loss.png" in result["plots"]


class TestGenericVariableNames:
    """Test that generic variable names trigger the manual-DAG-building path."""

    @pytest.mark.asyncio
    async def test_generic_names_return_empty_dag(self, orchestrator):
        """x1, x2, x3 are generic — should NOT call the LLM, should return empty DAG."""
        result = await orchestrator._execute_tool("propose_dag", {"vars": ["x1", "x2", "x3"]})

        assert result["success"]
        assert result["data"]["edges"] == []
        assert result["data"]["_generic_names"] is True
        assert "Cannot infer" in result["data"]["explanation"]

    @pytest.mark.asyncio
    async def test_generic_names_still_set_dag_proposed(self, orchestrator):
        """Even with generic names, state should move to dag_proposed (empty DAG)."""
        await orchestrator._execute_tool("propose_dag", {"vars": ["x1", "x2", "x3"]})

        assert orchestrator.state.current_step == "dag_proposed"
        assert orchestrator.state.proposed_dag is not None
        assert orchestrator.state.proposed_dag["edges"] == []

    @pytest.mark.asyncio
    async def test_generic_names_format_tells_user(self, orchestrator):
        """_format_tool_results should tell user to build DAG manually."""
        result = await orchestrator._execute_tool("propose_dag", {"vars": ["col1", "col2"]})
        results = [("propose_dag", result)]
        text = await orchestrator._format_tool_results(results, "propose a dag")

        assert "Cannot automatically infer" in text
        assert "generic" in text.lower()
        assert "open dag editor" in text.lower()

    @pytest.mark.asyncio
    async def test_meaningful_names_call_llm(self, orchestrator, mock_llm):
        """Descriptive names (age, income, …) should call the LLM."""
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            "age,income\nincome,spending"
        )
        # Replace generic data with meaningful names
        orchestrator.state.data_df = None  # Avoid fallback to x1/x2/x3

        result = await orchestrator._execute_tool(
            "propose_dag", {"vars": ["age", "income", "spending"]}
        )
        assert result["success"]
        assert result["data"].get("_generic_names") is not True
        assert len(result["data"]["edges"]) > 0

    @pytest.mark.asyncio
    async def test_mixed_names_treated_as_meaningful(self, orchestrator, mock_llm):
        """If at least one name is descriptive, treat the whole set as meaningful."""
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            "x1,blood_pressure"
        )
        orchestrator.state.data_df = None

        result = await orchestrator._execute_tool(
            "propose_dag", {"vars": ["x1", "blood_pressure"]}
        )
        assert result["success"]
        assert result["data"].get("_generic_names") is not True


# ============================================================================
# Agent Loop (process_message)
# ============================================================================

class TestAgentLoop:
    """Test the full agent loop: message → LLM → tool → response."""

    @pytest.mark.asyncio
    async def test_agent_returns_text_when_no_tool_call(self, orchestrator, mock_llm):
        """When LLM doesn't select a tool, return its text response."""
        response = await orchestrator.process_message("hello")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_agent_calls_tool_and_formats_response(
        self, orchestrator, mock_llm_with_tool_call, sample_dataframe
    ):
        """When LLM selects show_associations, response should contain correlation."""
        orchestrator.state.data_df = sample_dataframe
        mock_llm_with_tool_call.make_tool_response("show_associations", {})

        response = await orchestrator.process_message("show associations")
        assert "Correlation" in response or "correlation" in response

    @pytest.mark.asyncio
    async def test_agent_calls_open_dag_editor(self, orchestrator, mock_llm_with_tool_call):
        """When LLM selects open_dag_editor, response should contain the HTML marker."""
        mock_llm_with_tool_call.make_tool_response("open_dag_editor", {})

        response = await orchestrator.process_message("open dag editor")
        assert "OPEN_DAG_EDITOR" in response

    @pytest.mark.asyncio
    async def test_agent_falls_back_on_failure(self, orchestrator, mock_llm):
        """If the agent loop throws, should fall back to workflow routing."""
        # Make the LLM call raise an exception
        mock_llm.chat.completions.create.side_effect = Exception("API error")

        # This should fall back to _route_by_workflow_step
        response = await orchestrator.process_message("hello")
        assert isinstance(response, str)
        assert len(response) > 0


class TestGuidedParameterIntake:
    """Guided prompt flow: missing args -> pending state -> completion."""

    @pytest.mark.asyncio
    async def test_compute_ate_missing_x_prompts(self, orchestrator_with_dag):
        orchestrator_with_dag.state.experiment_dir = "experiments/test-guided"
        result = await orchestrator_with_dag._execute_tool("compute_ate", {"Y": "x3"})
        assert result["success"] is True
        assert "_guided_prompt" in result["data"]
        assert orchestrator_with_dag.state.pending_tool == "compute_ate"
        assert orchestrator_with_dag.state.pending_missing_param == "X"

    @pytest.mark.asyncio
    async def test_pending_completion_runs_tool(self, orchestrator_with_dag):
        orchestrator_with_dag.state.experiment_dir = "experiments/test-guided"
        orchestrator_with_dag._set_pending_tool("compute_ate", {"Y": "x3"}, "X")

        with patch.object(CausalTools, "compute_ate", return_value={
            "ate": 0.42,
            "x_treated": 1.0,
            "x_control": 0.0,
            "y_treated_mean": 1.2,
            "y_treated_std": 0.1,
            "y_control_mean": 0.78,
            "y_control_std": 0.1,
        }):
            response = await orchestrator_with_dag._try_handle_pending_tool("x1")

        assert response is not None
        assert "ATE" in response
        # After selecting X, current flow asks user to reconfirm variables before running ATE.
        assert orchestrator_with_dag.state.pending_tool == "compute_ate"
        assert orchestrator_with_dag.state.pending_missing_param == "X"

    @pytest.mark.asyncio
    async def test_pending_does_not_block_free_form_jump(
        self, orchestrator, mock_llm_with_tool_call, sample_dataframe
    ):
        orchestrator.state.experiment_dir = "experiments/test-guided"
        orchestrator.state.pending_tool = "compute_ate"
        orchestrator.state.pending_tool_args = {"Y": "x3"}
        orchestrator.state.pending_missing_param = "X"
        orchestrator.state.data_df = sample_dataframe
        mock_llm_with_tool_call.make_tool_response("show_associations", {})

        response = await orchestrator.process_message("show associations")
        assert "Correlation" in response or "correlation" in response

    @pytest.mark.asyncio
    async def test_stale_pending_fit_model_is_cleared(self, orchestrator_with_dag):
        """At dag_proposed, fit pending should continue guided defaults (not reroute)."""
        orchestrator_with_dag.state.current_step = "dag_proposed"
        orchestrator_with_dag.state.pending_tool = "fit_model"
        orchestrator_with_dag.state.pending_tool_args = {"epochs": 100}
        orchestrator_with_dag.state.pending_missing_param = "learning_rate"

        response = await orchestrator_with_dag._try_handle_pending_tool("default")
        assert response is not None
        assert orchestrator_with_dag.state.pending_tool == "fit_model"
        assert orchestrator_with_dag.state.pending_missing_param == "batch_size"


# ============================================================================
# _format_tool_results
# ============================================================================

class TestFormatToolResults:
    """Test that tool results are formatted into proper markdown."""

    @pytest.mark.asyncio
    async def test_format_propose_dag_has_edges(self, orchestrator):
        results = [("propose_dag", {
            "success": True,
            "data": {
                "variables": ["x1", "x2", "x3"],
                "edges": [("x1", "x2"), ("x2", "x3")],
                "explanation": "x1 causes x2 causes x3",
            },
            "plots": [],
            "implied_cis": ["x1 \u27C2 x3 | x2"],
        })]
        text = await orchestrator._format_tool_results(results, "propose a dag")
        assert "x1" in text
        assert "x2" in text
        assert "\u2192" in text or "->" in text  # Arrow in edges

    @pytest.mark.asyncio
    async def test_format_error_shows_message(self, orchestrator):
        results = [("test_dag", {
            "success": False,
            "error": "DAG or data not available.",
            "data": None,
            "plots": [],
        })]
        text = await orchestrator._format_tool_results(results, "test")
        assert "Error" in text
        assert "DAG" in text

    @pytest.mark.asyncio
    async def test_format_show_associations_has_matrix(self, orchestrator):
        results = [("show_associations", {
            "success": True,
            "data": {
                "correlation_matrix": "     x1    x2\nx1  1.00  0.50\nx2  0.50  1.00",
                "variables": ["x1", "x2"],
            },
            "plots": [],
        })]
        text = await orchestrator._format_tool_results(results, "show associations")
        assert "Correlation" in text
        assert "x1" in text

    @pytest.mark.asyncio
    async def test_format_open_dag_editor_has_marker(self, orchestrator):
        results = [("open_dag_editor", {
            "success": True,
            "data": {"action": "open_dag_editor"},
            "plots": [],
        })]
        text = await orchestrator._format_tool_results(results, "open editor")
        assert "OPEN_DAG_EDITOR" in text

    @pytest.mark.asyncio
    async def test_format_compute_ate_has_numbers(self, orchestrator):
        results = [("compute_ate", {
            "success": True,
            "data": {
                "X": "x1", "Y": "x3",
                "ate": 0.5432,
                "x_treated": 1.0, "x_control": 0.0,
                "y_treated_mean": 1.2, "y_treated_std": 0.3,
                "y_control_mean": 0.7, "y_control_std": 0.2,
            },
            "plots": [],
        })]
        text = await orchestrator._format_tool_results(results, "effect of x1 on x3")
        assert "ATE" in text
        assert "0.5432" in text

    @pytest.mark.asyncio
    async def test_format_generate_report(self, orchestrator):
        results = [("generate_report", {
            "success": True,
            "data": {"report_path": "/tmp/test.pdf", "report_type": "full"},
            "plots": [],
        })]
        text = await orchestrator._format_tool_results(results, "generate report")
        assert "Report" in text

    @pytest.mark.asyncio
    async def test_format_create_plots_with_urls(self, orchestrator):
        results = [("create_plots", {
            "success": True,
            "data": {"plot_types": ["all"], "num_plots": 2},
            "plots": ["/api/plots/test1.png", "/api/plots/test2.png"],
        })]
        text = await orchestrator._format_tool_results(results, "show plots")
        assert "2" in text
        assert "/api/plots/" in text


# ============================================================================
# Deterministic Pre-Routing
# ============================================================================

class TestDeterministicPreRouting:
    """
    Test _try_deterministic_route — intercepts workflow commands
    before the agent loop so they go to the correct handler.
    """

    def test_apply_revisions_is_intercepted(self, orchestrator):
        """'apply revisions' should be caught by the pre-router."""
        result = orchestrator._try_deterministic_route("apply revisions")
        assert result is not None  # Returns a coroutine

    def test_suggest_revisions_is_intercepted(self, orchestrator):
        result = orchestrator._try_deterministic_route("suggest revisions")
        assert result is not None

    def test_proceed_with_fitting_is_intercepted(self, orchestrator):
        result = orchestrator._try_deterministic_route("proceed with fitting")
        assert result is not None

    def test_normal_message_not_intercepted(self, orchestrator):
        result = orchestrator._try_deterministic_route("hello, what can you do?")
        assert result is None

    def test_open_dag_editor_not_intercepted(self, orchestrator):
        """'open dag editor' should be deterministically intercepted."""
        result = orchestrator._try_deterministic_route("open dag editor")
        assert result is not None

    def test_fit_model_not_intercepted(self, orchestrator):
        """Direct fit request should be deterministically intercepted when ready."""
        orchestrator.state.proposed_dag = {
            "variables": ["x1", "x2"],
            "edges": [("x1", "x2")],
            "adjacency_matrix": [[0, 1], [0, 0]],
        }
        orchestrator.state.data_path = "/tmp/test.csv"
        result = orchestrator._try_deterministic_route("fit model")
        assert result is not None

    def test_default_at_dag_proposed_is_intercepted_for_test(self, orchestrator_with_dag):
        orchestrator_with_dag.state.current_step = "dag_proposed"
        result = orchestrator_with_dag._try_deterministic_route("default")
        assert result is not None

    @pytest.mark.asyncio
    async def test_apply_revisions_calls_handler(self, orchestrator_with_dag, mock_llm):
        """Full integration: 'apply revisions' goes through process_message
        and reaches _apply_revisions_to_dag (which needs CI results)."""
        orchestrator_with_dag.state.current_step = "dag_finalized"
        orchestrator_with_dag.state.ci_test_results = {
            "consistent": False,
            "rejected_count": 1,
            "tests": [
                {"ci": "x1 _||_ x3 | x2", "rejected": True, "p_value": 0.01}
            ],
        }
        # Mock the LLM to return valid revision JSON
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            '{"add_edges": [["x1", "x3"]], "remove_edges": []}'
        )

        response = await orchestrator_with_dag.process_message("apply revisions")
        assert "revisions applied" in response.lower() or "DAG revisions" in response
        # The DAG should have been updated
        edges = orchestrator_with_dag.state.proposed_dag["edges"]
        edge_set = {(p, c) for p, c in edges}
        assert ("x1", "x3") in edge_set

    @pytest.mark.asyncio
    async def test_suggest_revisions_calls_handler(self, orchestrator_with_dag, mock_llm):
        """'suggest revisions' should call _step_4_propose_revisions."""
        orchestrator_with_dag.state.current_step = "dag_tested"
        orchestrator_with_dag.state.ci_test_results = {
            "consistent": False,
            "rejected_count": 1,
            "tests": [
                {"ci": "x1 _||_ x3 | x2", "rejected": True, "p_value": 0.01}
            ],
        }
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            "I suggest adding an edge from x1 to x3."
        )

        response = await orchestrator_with_dag.process_message("suggest revisions")
        assert "Revision" in response or "revision" in response
