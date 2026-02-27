"""
Tests for CausalTools — the actual statistical tool implementations.

Tests that can run WITHOUT the full TRAM-DAG / R stack are marked as unit tests.
Tests requiring the model fitting pipeline are marked with @pytest.mark.slow.

Validates:
  - DAG layout computation
  - Implied conditional independence computation
  - DAG plot generation
  - Association / correlation computation
  - propose_dag_from_llm (with mocked LLM)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from .conftest import CausalTools, CHATBOT_DIR


# ============================================================================
# DAG Layout Computation
# ============================================================================

class TestComputeDagLayout:
    """Test CausalTools._compute_dag_layout — pure computation, no deps."""

    def test_layout_returns_dict(self, sample_dag):
        import networkx as nx

        G = nx.DiGraph()
        for v in sample_dag["variables"]:
            G.add_node(v)
        for p, c in sample_dag["edges"]:
            G.add_edge(p, c)

        pos = CausalTools._compute_dag_layout(G, sample_dag["variables"])
        assert isinstance(pos, dict)
        assert set(pos.keys()) == set(sample_dag["variables"])

    def test_layout_positions_are_2d(self, sample_dag):
        import networkx as nx

        G = nx.DiGraph()
        for v in sample_dag["variables"]:
            G.add_node(v)
        for p, c in sample_dag["edges"]:
            G.add_edge(p, c)

        pos = CausalTools._compute_dag_layout(G, sample_dag["variables"])
        for var, (x, y) in pos.items():
            assert isinstance(x, (int, float)), f"x for {var} is not numeric"
            assert isinstance(y, (int, float)), f"y for {var} is not numeric"


# ============================================================================
# Implied Conditional Independencies
# ============================================================================

class TestComputeImpliedCIs:
    """Test CausalTools._compute_implied_cis — pure DAG computation."""

    def test_chain_dag_has_one_ci(self, sample_dag):
        """x1 → x2 → x3 implies x1 ⊥ x3 | x2."""
        cis = CausalTools._compute_implied_cis(sample_dag)
        assert isinstance(cis, list)
        # A chain x1->x2->x3 implies x1 ⊥ x3 | x2
        assert len(cis) >= 1
        # Find the CI involving x1 and x3
        found = False
        for ci in cis:
            pair = {ci["x"], ci["y"]}
            if pair == {"x1", "x3"}:
                found = True
                assert "x2" in ci["conditioning_set"]
        assert found, "Expected x1 ⊥ x3 | x2 not found"

    def test_fully_connected_dag_has_no_ci(self, sample_dag_full):
        """x1 → x2, x1 → x3, x2 → x3 has no implied CIs (fully connected)."""
        cis = CausalTools._compute_implied_cis(sample_dag_full)
        assert isinstance(cis, list)
        assert len(cis) == 0, f"Expected 0 CIs for fully connected DAG, got {len(cis)}"

    def test_empty_dag_returns_list(self, sample_variables):
        """DAG with no edges: should return a list (may be empty if only
        conditional — not marginal — independencies are computed)."""
        dag = {
            "adjacency_matrix": np.zeros((3, 3), dtype=int).tolist(),
            "variables": sample_variables,
            "edges": [],
        }
        cis = CausalTools._compute_implied_cis(dag)
        assert isinstance(cis, list)


# ============================================================================
# DAG Plot Generation
# ============================================================================

class TestDagPlotGeneration:
    """Test CausalTools._create_dag_plot — generates a PNG file."""

    def test_creates_png_file(self, sample_dag):
        session_id = "test-plot-session"
        result = CausalTools._create_dag_plot(session_id, sample_dag)
        assert result is not None
        assert Path(result).exists()
        assert Path(result).suffix == ".png"

    def test_plot_with_ci_returns_tuple(self, sample_dag):
        session_id = "test-plot-ci"
        result = CausalTools._create_dag_plot_with_ci(session_id, sample_dag)
        if result is not None:
            img_path, implied_cis = result
            assert Path(img_path).exists()
            assert isinstance(implied_cis, list)


# ============================================================================
# Association / Correlation
# ============================================================================

class TestAssociations:
    """Test computing correlations from data (no CausalTools method — done inline)."""

    def test_correlation_matrix_shape(self, sample_dataframe):
        corr = sample_dataframe.corr()
        n_vars = sample_dataframe.shape[1]
        assert corr.shape == (n_vars, n_vars)

    def test_correlation_diagonal_is_one(self, sample_dataframe):
        corr = sample_dataframe.corr()
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-10

    def test_correlation_values_in_range(self, sample_dataframe):
        corr = sample_dataframe.corr()
        assert (corr >= -1.0).all().all()
        assert (corr <= 1.0).all().all()


# ============================================================================
# propose_dag_from_llm (with mocked LLM)
# ============================================================================

class TestProposeDag:
    """Test CausalTools.propose_dag_from_llm with mocked LLM."""

    def test_returns_dag_structure(self, sample_variables, mock_llm):
        """Mock the LLM to return edges, verify DAG structure is built."""
        # Configure mock to return valid edge CSV
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            "x1,x2\nx2,x3"
        )

        dag = CausalTools.propose_dag_from_llm(sample_variables)

        assert "variables" in dag
        assert "edges" in dag
        assert "adjacency_matrix" in dag
        assert set(dag["variables"]) == set(sample_variables)
        assert len(dag["edges"]) == 2

    def test_handles_empty_llm_response(self, sample_variables, mock_llm):
        """If LLM returns no edges, fallback edges (if any) must still be valid."""
        mock_llm.chat.completions.create.return_value.choices[0].message.content = ""

        dag = CausalTools.propose_dag_from_llm(sample_variables)

        assert dag["variables"] == sample_variables
        for parent, child in dag["edges"]:
            assert parent in sample_variables
            assert child in sample_variables

    def test_handles_invalid_variable_names(self, sample_variables, mock_llm):
        """Edges referencing non-existent variables should be ignored."""
        mock_llm.chat.completions.create.return_value.choices[0].message.content = (
            "x1,x2\nfoo,bar\nx2,x3"
        )

        dag = CausalTools.propose_dag_from_llm(sample_variables)

        # foo,bar should be ignored
        valid_edges = dag["edges"]
        for p, c in valid_edges:
            assert p in sample_variables
            assert c in sample_variables


# ============================================================================
# Loss Plot Generation (requires experiment dir with data)
# ============================================================================

class TestLossPlot:
    """Test loss plot generation using existing experiment data."""

    def _find_experiment_dir(self):
        """Find an existing experiment dir with loss data for testing."""
        exp_root = CHATBOT_DIR / "experiments"
        if not exp_root.exists():
            return None
        for d in exp_root.iterdir():
            if d.is_dir():
                # Check if it has loss history files
                for sub in d.iterdir():
                    if sub.is_dir():
                        loss_file = sub / "train_loss_hist.json"
                        if loss_file.exists():
                            return str(d)
        return None

    def test_loss_plot_from_existing_experiment(self):
        exp_dir = self._find_experiment_dir()
        if exp_dir is None:
            pytest.skip("No experiment directory with loss data found")

        result = CausalTools._create_loss_plot(exp_dir, session_id="test-loss")
        assert result is not None
        assert Path(result).exists()


class TestMCPArtifactShaping:
    """Validate artifact metadata shaping from MCP responses."""

    def test_sample_from_model_include_metadata(self):
        with patch.object(CausalTools, "_call_mcp", return_value={
            "success": True,
            "data": {
                "samples": {"x1": {"values": [1.0, 2.0], "summary": {"mean": 1.5, "std": 0.5, "min": 1.0, "max": 2.0}}},
                "artifact_manifest": [{"kind": "sampling_distributions", "path": "/tmp/sampling_distributions.png"}],
            },
            "artifacts": ["/tmp/sampling_distributions.png"],
            "tables": {},
            "messages": [],
            "error": None,
        }):
            result = CausalTools.sample_from_model(
                experiment_dir="/tmp/exp",
                n_samples=1000,
                include_metadata=True,
            )
        assert "samples" in result
        assert "mcp_artifacts" in result
        assert "artifact_manifest" in result

    def test_fit_model_includes_mcp_artifacts(self):
        with patch.object(CausalTools, "_call_mcp", return_value={
            "success": True,
            "data": {
                "experiment_dir": "/tmp/exp",
                "config_path": "/tmp/exp/configuration.json",
                "loss_history": {},
                "artifact_manifest": [{"kind": "loss_history", "path": "/tmp/loss_history.png"}],
            },
            "artifacts": ["/tmp/loss_history.png"],
            "tables": {},
            "messages": [],
            "error": None,
        }):
            result = CausalTools.fit_tramdag_model(
                dag={"adjacency_matrix": [[0, 1], [0, 0]], "variables": ["x1", "x2"], "edges": [("x1", "x2")]},
                data_path="/tmp/data.csv",
                experiment_dir="/tmp/exp",
            )
        assert "mcp_artifacts" in result
        assert result["mcp_artifacts"] == ["/tmp/loss_history.png"]
