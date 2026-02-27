from __future__ import annotations

from typing import Any, Dict


# Session storage (in production, use Redis or database)
sessions: Dict[str, Dict[str, Any]] = {}


def invalidate_session_after_dag_update(session: Dict[str, Any]) -> None:
    """Clear derived artifacts/results after DAG structure changes."""
    session["fitted_model"] = None
    session["experiment_dir"] = None
    session["ci_test_results"] = None
    session["query_results"] = {}
    session["mcp_plot_urls_by_kind"] = {}
    session["pending_tool"] = None
    session["pending_tool_args"] = None
    session["pending_missing_param"] = None
