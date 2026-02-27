class ToolRegistry:
    """
    Registry of all tools available to the AI Agent.
    """

    TOOLS = [
        {
            "name": "propose_dag",
            "description": (
                "Propose a causal DAG structure based on variable names. "
                "Call this when the user wants to create or propose a DAG. "
                "Uses LLM reasoning to suggest plausible causal edges."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variable names from the dataset"
                    },
                    "expert_text": {
                        "type": "string",
                        "description": "Optional domain knowledge about causal relationships"
                    }
                },
                "required": ["vars"]
            }
        },
        {
            "name": "test_dag",
            "description": (
                "Test if a DAG is consistent with observed data using conditional "
                "independence (CI) tests. Call this when the user wants to test/validate "
                "their DAG, or says 'yes, test the model'. Runs locally using R."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "alpha": {
                        "type": "number",
                        "description": "Significance level for CI tests (default 0.05)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "fit_model",
            "description": (
                "Fit a TRAM-DAG model to the data using the current DAG structure. "
                "Call this when the user wants to fit/train the model. "
                "Takes a few minutes. All computation runs locally (PyTorch)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "epochs": {
                        "type": "integer",
                        "description": "Number of training epochs (default 100)"
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Learning rate (default 0.01)"
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size (default 512)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "sample",
            "description": (
                "Sample data points from the fitted TRAM-DAG model. "
                "Can do observational sampling or interventional sampling (do-calculus). "
                "Call this when user asks to sample, or asks 'what if X = value'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n_samples": {
                        "type": "integer",
                        "description": "Number of samples to generate (default 10000)"
                    },
                    "do_interventions": {
                        "type": "object",
                        "description": "Interventions as {variable: value}, e.g. {'x1': 1.0}"
                    }
                },
                "required": []
            }
        },
        {
            "name": "compute_ate",
            "description": (
                "Compute Average Treatment Effect (ATE) between a treatment and outcome variable. "
                "Call this when user asks about causal effects, e.g. 'effect of X on Y'. "
                "Uses do-calculus: ATE = E[Y|do(X=treated)] - E[Y|do(X=control)]."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "X": {"type": "string", "description": "Treatment variable name"},
                    "Y": {"type": "string", "description": "Outcome variable name"},
                    "x_treated": {"type": "number", "description": "Treatment value (default 1.0)"},
                    "x_control": {"type": "number", "description": "Control value (default 0.0)"},
                    "n_samples": {"type": "integer", "description": "Number of samples for estimation (default 10000)"}
                },
                "required": ["X", "Y"]
            }
        },
        {
            "name": "create_plots",
            "description": (
                "Generate visualizations: sampling distributions, loss history, "
                "DAG structure, or intervention plots. Call this when the user asks "
                "to see plots, visualizations, graphs, loss history, or distributions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["distributions", "loss", "dag", "intervention", "samples_vs_true", "linear_shift_history", "simple_intercepts_history", "hdag", "latents", "all"]},
                        "description": "Which plots to generate. Use 'all' for everything."
                    },
                    "n_samples": {
                        "type": "integer",
                        "description": "Number of samples for distribution plots (default 10000)"
                    }
                },
                "required": ["plot_types"]
            }
        },
        {
            "name": "show_associations",
            "description": (
                "Compute and display the correlation matrix from the uploaded data. "
                "Call this when user asks about associations, correlations, or "
                "relationships between variables."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        },
        {
            "name": "generate_report",
            "description": (
                "Generate a comprehensive PDF report with all results: DAG, "
                "training loss, distributions, interventions, and analysis summary. "
                "Call this when user asks to download or generate a report."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "enum": ["full", "intervention"],
                        "description": "Report type: 'full' or 'intervention' (default 'full')"
                    }
                },
                "required": []
            }
        },
        {
            "name": "open_dag_editor",
            "description": (
                "Open the interactive visual DAG editor in the chat UI. "
                "Call this when the user wants to visually build or edit their DAG, "
                "or says 'open dag editor'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    ]

    @staticmethod
    def get_llm_tool_schemas() -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            for tool in ToolRegistry.TOOLS
        ]

    @staticmethod
    def get_tool_names() -> list:
        return [tool["name"] for tool in ToolRegistry.TOOLS]

    @staticmethod
    def get_tool_descriptions() -> str:
        lines = []
        for i, tool in enumerate(ToolRegistry.TOOLS, 1):
            lines.append(f"  {i}. {tool['name']:20s} — {tool['description'][:80]}")
        return "\n".join(lines)
