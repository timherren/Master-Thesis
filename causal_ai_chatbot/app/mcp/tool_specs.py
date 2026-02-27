"""
Unified MCP tool contract for cross-app execution.

This file defines a normalized schema the chatbot can rely on,
independent of whether implementation is R or Python.
"""

MCP_RESULT_SCHEMA = {
    "success": "bool",
    "data": "object",
    "artifacts": "string[]",
    "tables": "object",
    "messages": "string[]",
    "error": "string|null",
}

MCP_TOOL_SPECS = {
    "propose_dag": {
        "backend": "causal_ai_chatbot/app/mcp_wrappers/dag_validator_wrapper.R",
        "payload": {"vars": "string[]", "expert_text": "string?"},
    },
    "test_dag": {
        "backend": "causal_ai_chatbot/app/mcp_wrappers/dag_validator_wrapper.R",
        "payload": {
            "dag": "object",
            "data_path": "string",
            "alpha": "number?",
            "tests": "string[]?",
        },
    },
    "fit_model": {
        "backend": "causal_ai_chatbot/app/mcp_wrappers/tram_wrapper.py",
        "payload": {
            "dag": "object",
            "data_path": "string",
            "experiment_dir": "string",
            "epochs": "integer?",
            "learning_rate": "number?",
            "batch_size": "integer?",
            "random_seed": "integer?",
        },
    },
    "sample": {
        "backend": "causal_ai_chatbot/app/mcp_wrappers/tram_wrapper.py",
        "payload": {
            "experiment_dir": "string",
            "n_samples": "integer?",
            "do_interventions": "object?",
            "random_seed": "integer?",
        },
    },
    "compute_ate": {
        "backend": "causal_ai_chatbot/app/mcp_wrappers/tram_wrapper.py",
        "payload": {
            "experiment_dir": "string",
            "X": "string",
            "Y": "string",
            "x_treated": "number?",
            "x_control": "number?",
            "n_samples": "integer?",
            "random_seed": "integer?",
        },
    },
}

