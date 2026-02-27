"""
Causal AI Agent Chatbot Server
FastAPI backend for conversational causal inference using TRAM-DAG

================================================================================
ARCHITECTURE OVERVIEW — AI Agent with Local Statistical Tools
================================================================================

The chatbot is an AI AGENT (LLM with function calling) that has access to a
set of LOCAL statistical tools. The agent decides WHICH tool to call based on
the user's natural-language message. All heavy computation runs locally.

    ┌─────────────┐     natural language      ┌──────────────────┐
    │   Chat UI   │ ──────────────────────────>│   AI Agent       │
    │  (browser)  │                            │  (Local LLM)     │
    │             │<──────────────────────────-│                  │
    └─────────────┘     formatted response     │  Decides which   │
                                               │  tool to call    │
                                               │  via function    │
                                               │  calling         │
                                               └────────┬─────────┘
                                                        │ tool call
                                                        ▼
                                               ┌──────────────────┐
                                               │  Tool Registry   │
                                               │  (LOCAL tools)   │
                                               │                  │
                                               │  1. propose_dag  │
                                               │  2. test_dag     │
                                               │  3. fit_model    │
                                               │  4. sample       │
                                               │  5. compute_ate  │
                                               │  6. create_plots │
                                               │  7. show_assoc.  │
                                               │  8. gen_report   │
                                               └──────────────────┘

PRIVACY: No user data leaves your machine for LLM reasoning. Only variable names, tool results
summaries, and agent decisions use the LLM. All statistical computation
(CI tests, TRAM-DAG fitting, sampling, ATE) runs locally.
================================================================================
"""

import os
import json
import uuid
import re
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime
import io
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
import sys

# PDF and plotting
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
# Ensure backend is set correctly
print(f"[DEBUG] Matplotlib backend: {matplotlib.get_backend()}")
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import networkx as nx

import pandas as pd
import numpy as np
from .models import ChatMessage, SessionState, ChatRequest, SaveDagRequest
from .tool_registry import ToolRegistry
from .session_store import sessions, invalidate_session_after_dag_update
from .runtime_context import (
    TramDagConfig,
    TramDagModel,
    TramDagDataset,
    llm_client,
    llm_model_decision,
    llm_model_interpretation,
    r_checker,
    mcp_client,
    plot_file_handles,
)
from .config import (
    APP_DIR,
    UPLOADS_DIR,
    REPORTS_DIR,
    TEMP_PLOTS_DIR,
    EXPERIMENTS_BASE_DIR,
)

app = FastAPI(title="Causal AI Agent Chatbot")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to keep file handles open and prevent deletion
# This ensures files persist even if there's process isolation or cleanup issues
_plot_file_handles = plot_file_handles


# ============================================================================
# TOOL IMPLEMENTATIONS — The actual statistical / plotting functions
# ============================================================================

class CausalTools:
    """
    Implementation of all tools the AI Agent can call.
    
    Each method here corresponds to a tool in ToolRegistry.TOOLS.
    All statistical computation runs locally — no data leaves the server.
    """

    @staticmethod
    def _mcp_available() -> bool:
        return mcp_client is not None and mcp_client.is_ready()

    @staticmethod
    def _call_mcp(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not CausalTools._mcp_available():
            return {"success": False, "error": "MCP backend unavailable"}
        res = mcp_client.call(tool_name, payload)
        return {
            "success": res.success,
            "data": res.data,
            "artifacts": res.artifacts,
            "tables": res.tables,
            "messages": res.messages,
            "error": res.error,
        }

    @staticmethod
    def import_mcp_artifacts(session_id: str, artifact_paths: List[str]) -> List[str]:
        """
        Copy MCP-produced artifacts into chatbot temp_plots and return API URLs.
        """
        urls: List[str] = []
        for src in artifact_paths or []:
            try:
                src_path = Path(src)
                if not src_path.exists():
                    continue
                target_name = f"{session_id}_{src_path.name}"
                target_path = TEMP_PLOTS_DIR / target_name
                shutil.copy2(src_path, target_path)
                urls.append(f"/api/plots/{target_name}")
            except Exception as exc:
                print(f"[MCP] Failed to import artifact {src}: {exc}")
        return urls
    
    @staticmethod
    def propose_dag_from_llm(vars: List[str], expert_text: Optional[str] = None) -> Dict:
        """Propose DAG strictly via local MCP wrapper."""
        mcp = CausalTools._call_mcp("propose_dag", {"vars": vars, "expert_text": expert_text})
        if not mcp.get("success"):
            raise RuntimeError(f"MCP propose_dag failed: {mcp.get('error', 'unknown error')}")
        dag_data = mcp.get("data", {})
        edges_df = dag_data.get("edges", [])
        if isinstance(edges_df, list) and edges_df and isinstance(edges_df[0], dict):
            edges = [(row.get("from"), row.get("to")) for row in edges_df]
        else:
            edges = dag_data.get("edges", [])
        return {
            "adjacency_matrix": dag_data.get("adjacency_matrix", []),
            "variables": dag_data.get("variables", vars),
            "edges": edges,
            "llm_explanation": dag_data.get("llm_explanation", ""),
            "mcp_artifacts": mcp.get("artifacts", []),
        }
    
    @staticmethod
    def test_dag_consistency(dag: Dict, data_path: str, alpha: float = 0.05) -> Dict:
        """Test DAG consistency strictly via local MCP wrapper."""
        mcp = CausalTools._call_mcp(
            "test_dag",
            {"dag": dag, "data_path": data_path, "alpha": alpha, "tests": ["gcm", "pcm"]},
        )
        if not mcp.get("success"):
            raise RuntimeError(f"MCP test_dag failed: {mcp.get('error', 'unknown error')}")
        data = mcp.get("data", {})
        data["mcp_artifacts"] = mcp.get("artifacts", [])
        data["mcp_tables"] = mcp.get("tables", {})
        return data
    
    @staticmethod
    def fit_tramdag_model(
        dag: Dict,
        data_path: str,
        experiment_dir: str,
        epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: int = 512,
        random_seed: int = 42,
    ) -> Dict:
        """Fit TRAM-DAG model strictly via local MCP wrapper."""
        mcp = CausalTools._call_mcp(
            "fit_model",
            {
                "dag": dag,
                "data_path": data_path,
                "experiment_dir": experiment_dir,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "random_seed": random_seed,
            },
        )
        if not mcp.get("success"):
            err = mcp.get("error", "unknown error")
            debug_tail = ""
            msgs = mcp.get("messages") or []
            if msgs:
                debug_tail = f"\nWrapper details:\n{msgs[-1]}"
            raise RuntimeError(f"MCP fit_model failed: {err}{debug_tail}")
        data = mcp.get("data", {})
        return {
            "experiment_dir": data.get("experiment_dir", experiment_dir),
            "config_path": data.get("config_path"),
            "model": None,
            "loss_history": data.get("loss_history", {}),
            "random_seed": data.get("random_seed", random_seed),
            "split_paths": data.get("split_paths", {}),
            "artifact_manifest": data.get("artifact_manifest", []),
            "mcp_artifacts": mcp.get("artifacts", []),
        }
    
    @staticmethod
    def sample_from_model(
        experiment_dir: str,
        n_samples: int = 10000,
        do_interventions: Optional[Dict[str, float]] = None,
        random_seed: int = 42,
        include_metadata: bool = False,
    ) -> Dict:
        """Sample from fitted TRAM-DAG model strictly via local MCP wrapper."""
        mcp = CausalTools._call_mcp(
            "sample",
            {
                "experiment_dir": experiment_dir,
                "n_samples": n_samples,
                "do_interventions": do_interventions or {},
                "random_seed": random_seed,
            },
        )
        if not mcp.get("success"):
            raise RuntimeError(f"MCP sample failed: {mcp.get('error', 'unknown error')}")
        data = mcp.get("data", {})
        if include_metadata:
            if "samples" not in data:
                data = {"samples": data}
            data["mcp_artifacts"] = mcp.get("artifacts", [])
            data["artifact_manifest"] = data.get("artifact_manifest", [])
            data["random_seed"] = data.get("random_seed", random_seed)
            return data
        if "samples" in data:
            return data.get("samples", {})
        return data
    
    @staticmethod
    def compute_ate(
        experiment_dir: str,
        X: str,
        Y: str,
        x_treated: float = 1.0,
        x_control: float = 0.0,
        n_samples: int = 10000,
        random_seed: int = 42,
    ) -> Dict:
        """Compute ATE strictly via local MCP wrapper."""
        mcp = CausalTools._call_mcp(
            "compute_ate",
            {
                "experiment_dir": experiment_dir,
                "X": X,
                "Y": Y,
                "x_treated": x_treated,
                "x_control": x_control,
                "n_samples": n_samples,
                "random_seed": random_seed,
            },
        )
        if not mcp.get("success"):
            raise RuntimeError(f"MCP compute_ate failed: {mcp.get('error', 'unknown error')}")
        data = mcp.get("data", {})
        data["mcp_artifacts"] = mcp.get("artifacts", [])
        data["artifact_manifest"] = data.get("artifact_manifest", [])
        data["random_seed"] = data.get("random_seed", random_seed)
        return data
    
    @staticmethod
    def _generate_reproducible_code(session_id: str, session: Dict) -> str:
        """Generate Python code to reproduce the analysis"""
        code_lines = []
        code_lines.append("# Reproducible Analysis Code")
        code_lines.append("# Generated automatically from session analysis")
        code_lines.append("")
        code_lines.append("import pandas as pd")
        code_lines.append("import numpy as np")
        code_lines.append("from pathlib import Path")
        code_lines.append("from tramdag import TramDagConfig, TramDagDataset, TramDagModel")
        code_lines.append("")
        
        # Data loading
        data_path = session.get('data_path')
        if data_path:
            code_lines.append("# 1. Load Data")
            if data_path.endswith('.csv'):
                code_lines.append(f"df = pd.read_csv('{data_path}')")
            elif data_path.endswith(('.xlsx', '.xls')):
                code_lines.append(f"df = pd.read_excel('{data_path}')")
            else:
                code_lines.append(f"# Load your data file: {data_path}")
            code_lines.append("print(f'Data shape: {df.shape}')")
            code_lines.append("print(f'Variables: {list(df.columns)}')")
            code_lines.append("")
        
        # DAG structure
        proposed_dag = session.get('proposed_dag', {})
        if proposed_dag:
            code_lines.append("# 2. Define DAG Structure")
            vars = proposed_dag.get('variables', [])
            edges = proposed_dag.get('edges', [])
            
            code_lines.append(f"variables = {vars}")
            code_lines.append("")
            code_lines.append("# Adjacency matrix")
            code_lines.append(f"n = len(variables)")
            code_lines.append("adj_matrix = np.zeros((n, n), dtype=int)")
            
            # Create adjacency matrix code
            var_to_idx = {v: i for i, v in enumerate(vars)}
            for parent, child in edges:
                if parent in var_to_idx and child in var_to_idx:
                    code_lines.append(f"adj_matrix[{var_to_idx[parent]}, {var_to_idx[child]}] = 1")
            
            code_lines.append("")
            code_lines.append("# Create DAG configuration")
            code_lines.append("dag_config = {")
            code_lines.append("    'variables': variables,")
            code_lines.append("    'adjacency_matrix': adj_matrix.tolist(),")
            code_lines.append("    'edges': [")
            for parent, child in edges:
                code_lines.append(f"        ('{parent}', '{child}'),")
            code_lines.append("    ]")
            code_lines.append("}")
            code_lines.append("")
        
        # Model fitting
        experiment_dir = session.get('experiment_dir')
        if experiment_dir:
            code_lines.append("# 3. Fit TRAM-DAG Model")
            code_lines.append(f"experiment_dir = '{experiment_dir}'")
            code_lines.append("")
            code_lines.append("# Create configuration")
            code_lines.append("config = TramDagConfig.from_dict(dag_config)")
            code_lines.append("")
            code_lines.append("# Create dataset")
            code_lines.append("dataset = TramDagDataset(df, config)")
            code_lines.append("")
            code_lines.append("# Fit model")
            code_lines.append("model = TramDagModel(config, dataset)")
            code_lines.append("model.fit(")
            code_lines.append("    n_epochs=100,  # Adjust as needed")
            code_lines.append("    batch_size=256,")
            code_lines.append("    learning_rate=0.001")
            code_lines.append(")")
            code_lines.append("")
            code_lines.append("# Save model")
            code_lines.append("model.save(experiment_dir)")
            code_lines.append("")
        
        # Queries
        query_results = session.get('query_results', {})
        if query_results:
            code_lines.append("# 4. Perform Causal Queries")
            
            if 'ate' in query_results:
                ate_data = query_results['ate']
                X = ate_data.get('X', 'X')
                Y = ate_data.get('Y', 'Y')
                x_treated = ate_data.get('x_treated', 1.0)
                x_control = ate_data.get('x_control', 0.0)
                
                code_lines.append("")
                code_lines.append("# Compute Average Treatment Effect (ATE)")
                code_lines.append(f"# Treatment: do({X} = {x_treated})")
                code_lines.append(f"# Control: do({X} = {x_control})")
                code_lines.append(f"# Outcome: {Y}")
                code_lines.append("")
                code_lines.append("# Sample under treatment")
                code_lines.append(f"samples_treated, _ = model.sample(")
                code_lines.append(f"    do_interventions={{'{X}': {x_treated}}},")
                code_lines.append("    number_of_samples=10000")
                code_lines.append(")")
                code_lines.append(f"y_treated = samples_treated['{Y}'].values")
                code_lines.append("")
                code_lines.append("# Sample under control")
                code_lines.append(f"samples_control, _ = model.sample(")
                code_lines.append(f"    do_interventions={{'{X}': {x_control}}},")
                code_lines.append("    number_of_samples=10000")
                code_lines.append(")")
                code_lines.append(f"y_control = samples_control['{Y}'].values")
                code_lines.append("")
                code_lines.append("# Compute ATE")
                code_lines.append("ate = np.mean(y_treated) - np.mean(y_control)")
                code_lines.append("print(f'ATE = {ate:.4f}')")
                code_lines.append("")
        
        # Sampling
        if experiment_dir:
            code_lines.append("# 5. Sample from Model")
            code_lines.append("# Observational sampling")
            code_lines.append("samples_obs, latents = model.sample(number_of_samples=10000)")
            code_lines.append("")
            code_lines.append("# Interventional sampling (example)")
            code_lines.append("# samples_int, _ = model.sample(")
            code_lines.append("#     do_interventions={'variable_name': value},")
            code_lines.append("#     number_of_samples=10000")
            code_lines.append("# )")
            code_lines.append("")
        
        # CI Tests
        ci_results = session.get('ci_test_results')
        if ci_results:
            code_lines.append("# 6. DAG Consistency Tests")
            code_lines.append("# Run conditional independence tests using R")
            code_lines.append("# See experiment directory for R scripts")
            code_lines.append("")
        
        code_lines.append("# End of reproducible code")
        
        return '\n'.join(code_lines)

    @staticmethod
    def _api_plot_url_to_path(plot_ref: Optional[str]) -> Optional[Path]:
        """Resolve /api/plots/... URL or absolute path to a local Path."""
        if not plot_ref:
            return None
        try:
            ref = str(plot_ref).strip()
            if ref.startswith("/api/plots/"):
                name = ref.split("/api/plots/", 1)[1].split("?", 1)[0]
                p = TEMP_PLOTS_DIR / name
                return p if p.exists() else None
            p = Path(ref)
            return p if p.exists() else None
        except Exception:
            return None

    @staticmethod
    def _export_reproducibility_artifacts(session_id: str, session: Dict) -> Dict[str, str]:
        """
        Export reproducibility artifacts matching tram_dag_application naming:
        - scripts/reproduce_analysis.R + scripts/README.md
        - reproducible_package/data.csv + reproducible_package/run_complete_workflow.R
        - reproducible_package/r_requirements.txt + reproducible_package/python_requirements.txt
        """
        experiment_dir = session.get("experiment_dir")
        if not experiment_dir:
            return {}

        exp_path = Path(experiment_dir)
        exp_path.mkdir(parents=True, exist_ok=True)

        scripts_dir = exp_path / "scripts"
        package_dir = exp_path / "reproducible_package"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        package_dir.mkdir(parents=True, exist_ok=True)

        dag = session.get("proposed_dag", {}) or {}
        dag_vars = dag.get("variables", [])
        dag_edges = dag.get("edges", [])
        dag_vars_json = json.dumps(dag_vars)
        dag_edges_json = json.dumps(dag_edges)

        # 1) scripts/reproduce_analysis.R (tram-compatible name)
        reproduce_analysis_r = f"""# Reproducible Analysis Script
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Experiment: {exp_path.name}

cat("Running reproducible analysis for experiment: {exp_path.name}\\n")

workflow_path <- file.path("..", "reproducible_package", "run_complete_workflow.R")
if (!file.exists(workflow_path)) {{
  stop("Missing workflow script: ", workflow_path)
}}

cat("Sourcing:", workflow_path, "\\n")
source(workflow_path)
"""
        reproduce_script_path = scripts_dir / "reproduce_analysis.R"
        reproduce_script_path.write_text(reproduce_analysis_r, encoding="utf-8")

        scripts_readme = f"""# Reproducible Analysis Scripts

Experiment: `{exp_path.name}`
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Files
- `reproduce_analysis.R`: Re-runs analysis steps against this experiment setup.

## Usage
```bash
cd "{exp_path}"
Rscript scripts/reproduce_analysis.R
```
"""
        (scripts_dir / "README.md").write_text(scripts_readme, encoding="utf-8")

        # 2) Self-contained reproducible package
        data_df = None
        if session.get("data_df"):
            try:
                data_df = pd.DataFrame(session["data_df"])
            except Exception:
                data_df = None

        data_csv_path = package_dir / "data.csv"
        if data_df is not None and not data_df.empty:
            data_df.to_csv(data_csv_path, index=False)
        else:
            data_path = session.get("data_path")
            if data_path and Path(data_path).exists() and str(data_path).lower().endswith(".csv"):
                shutil.copy2(data_path, data_csv_path)

        # 2) reproducible_package/run_complete_workflow.R (tram-compatible name)
        full_workflow = f"""# ============================================================================
# Complete Reproducible TRAM-DAG Workflow
# ============================================================================
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# Experiment: {exp_path.name}
# ============================================================================

required_r_packages <- c("reticulate", "jsonlite")
missing_r <- required_r_packages[!sapply(required_r_packages, requireNamespace, quietly = TRUE)]
if (length(missing_r) > 0) {{
  install.packages(missing_r, repos = "https://cloud.r-project.org")
}}

library(reticulate)
library(jsonlite)

cat("\\n=== Step 1: Configure Python environment ===\\n")
tryCatch({{
  use_condaenv("tramdag", required = TRUE)
  cat("Using conda env: tramdag\\n")
}}, error = function(e) {{
  cat("Conda env 'tramdag' not found; using default Python.\\n")
}})

tramdag <- import("tramdag")
TramDagConfig <- tramdag$TramDagConfig
TramDagDataset <- tramdag$TramDagDataset
TramDagModel <- tramdag$TramDagModel

cat("\\n=== Step 2: Load data ===\\n")
data_file <- "data.csv"
if (!file.exists(data_file)) {{
  stop("data.csv not found in reproducible_package/")
}}
df <- read.csv(data_file, check.names = TRUE)
cat("Data shape:", nrow(df), "rows x", ncol(df), "cols\\n")

cat("\\n=== Step 3: Rebuild DAG ===\\n")
variables <- fromJSON('{dag_vars_json}')
edges <- fromJSON('{dag_edges_json}')
if (length(variables) == 0) {{
  variables <- colnames(df)
}}
n <- length(variables)
adj_matrix <- matrix(0L, nrow = n, ncol = n, dimnames = list(variables, variables))

if (length(edges) > 0) {{
  for (edge in edges) {{
    parent <- edge[[1]]
    child <- edge[[2]]
    if (!is.null(parent) && !is.null(child) && parent %in% variables && child %in% variables) {{
      adj_matrix[parent, child] <- 1L
    }}
  }}
}}

adjacency_list <- lapply(seq_len(nrow(adj_matrix)), function(i) as.integer(adj_matrix[i, ]))
dag_config <- r_to_py(list(
  variables = as.list(variables),
  adjacency_matrix = adjacency_list,
  edges = edges
))

cat("\\n=== Step 4: Fit model ===\\n")
config <- TramDagConfig$from_dict(dag_config)
dataset <- TramDagDataset(r_to_py(df), config)
model <- TramDagModel(config, dataset)

tryCatch({{
  model$fit(n_epochs = as.integer(100), learning_rate = 0.01, batch_size = as.integer(512))
}}, error = function(e) {{
  model$fit(epochs = as.integer(100), learning_rate = 0.01, batch_size = as.integer(512))
}})

out_dir <- "reproduced_experiment"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
model$save(out_dir)
cat("Model saved to:", out_dir, "\\n")
"""
        full_workflow_path = package_dir / "run_complete_workflow.R"
        full_workflow_path.write_text(full_workflow, encoding="utf-8")

        r_reqs_text = """reticulate
jsonlite
"""
        (package_dir / "r_requirements.txt").write_text(r_reqs_text, encoding="utf-8")

        reqs_text = """tramdag
numpy
pandas
torch
"""
        (package_dir / "python_requirements.txt").write_text(reqs_text, encoding="utf-8")

        package_readme = f"""# Reproducible Package

This folder is a portable export of the analysis context for experiment `{exp_path.name}`.

## Files
- `data.csv` (copied dataset when available)
- `run_complete_workflow.R` (fit-from-scratch workflow)
- `r_requirements.txt`
- `python_requirements.txt`

## Run
```bash
cd "{package_dir}"
Rscript run_complete_workflow.R
```

Or with explicit dependency install:
```bash
R -e "install.packages(scan('r_requirements.txt', what='character'))"
pip install -r python_requirements.txt
Rscript run_complete_workflow.R
```
"""
        (package_dir / "README.md").write_text(package_readme, encoding="utf-8")

        return {
            "scripts_dir": str(scripts_dir),
            "reproducible_package_dir": str(package_dir),
            "reproduce_script": str(reproduce_script_path),
            "full_workflow_script": str(full_workflow_path),
        }
    
    @staticmethod
    def generate_report(session_id: str, output_path: str, report_type: str = "full") -> str:
        """
        Generate comprehensive PDF report with plots
        
        Args:
            session_id: Session identifier
            output_path: Path to save PDF
            report_type: "full" (after model fitting) or "intervention" (after intervention query)
        """
        session = sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        story = []
        styles = getSampleStyleSheet()

        # Helper for consistent figure insertion.
        def add_plot(plot_path: Optional[Path], caption: str, width_in: float = 5.8, height_in: float = 3.6) -> bool:
            if not plot_path or not plot_path.exists():
                return False
            try:
                story.append(Image(str(plot_path), width=width_in * inch, height=height_in * inch))
                story.append(Paragraph(f"<i>{caption}</i>", styles["Normal"]))
                story.append(Spacer(1, 0.12 * inch))
                return True
            except Exception:
                return False

        # Title and metadata
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#1a1a1a"),
            spaceAfter=18,
            alignment=TA_CENTER,
        )
        story.append(Paragraph("Causal Inference Analysis Report", title_style))
        story.append(Paragraph(f"<b>Session ID:</b> {session_id}", styles["Normal"]))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        # 1) Executive Summary (tram-style wording)
        story.append(Paragraph("1. Executive Summary", styles["Heading2"]))
        user_question = session.get("user_question", "N/A")
        story.append(Paragraph(f"<b>Causal question:</b> {user_question}", styles["Normal"]))
        data_info = session.get("data_info", {}) or {}
        if data_info:
            shape = data_info.get("shape", ["?", "?"])
            cols = data_info.get("columns", [])
            summary_text = (
                "This report presents a causal inference analysis using TRAM-DAG "
                f"(Transformation Models for Directed Acyclic Graphs). A dataset with "
                f"<b>{shape[0]} observations</b> across <b>{shape[1]} variables</b> "
                f"({', '.join(cols)}) was analyzed."
            )
            story.append(Paragraph(summary_text, styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

        # 2) Data overview and pairwise relationships
        story.append(Paragraph("2. Data Overview", styles["Heading2"]))
        if data_info:
            shape = data_info.get("shape", ["?", "?"])
            cols = data_info.get("columns", [])
            story.append(Paragraph(f"<b>Number of observations:</b> {shape[0]}", styles["Normal"]))
            story.append(Paragraph(f"<b>Number of variables:</b> {shape[1]}", styles["Normal"]))
            story.append(Paragraph(f"<b>Variables:</b> {', '.join(cols)}", styles["Normal"]))
        story.append(Paragraph("2.1 Pairwise Relationships", styles["Heading3"]))
        story.append(
            Paragraph(
                "The pairplot below shows the pairwise scatter plots and marginal distributions for variables in the dataset.",
                styles["Normal"],
            )
        )

        pairplot_ref = session.get("data_pairplot_url")
        pairplot_path = CausalTools._api_plot_url_to_path(pairplot_ref)
        if not pairplot_path and session.get("data_df"):
            try:
                pairplot_ref = CausalTools._create_data_pairplot(session_id, pd.DataFrame(session.get("data_df")))
                pairplot_path = CausalTools._api_plot_url_to_path(pairplot_ref)
            except Exception:
                pairplot_path = None
        add_plot(pairplot_path, "Figure 1: Data pairplot with marginal distributions.", width_in=5.7, height_in=5.4)
        story.append(Spacer(1, 0.1 * inch))

        # 3) Causal DAG specification
        story.append(Paragraph("3. Causal DAG Specification", styles["Heading2"]))
        proposed_dag = session.get("proposed_dag", {}) or {}
        ci_results = session.get("ci_test_results")
        if proposed_dag:
            vars_list = proposed_dag.get("variables", [])
            story.append(Paragraph(f"<b>Variables in the DAG:</b> {', '.join(vars_list)}", styles["Normal"]))
            dag_path: Optional[Path] = None
            try:
                if ci_results:
                    dag_ci_path = CausalTools._create_dag_plot_with_ci_results(session_id, proposed_dag, ci_results)
                    dag_path = Path(dag_ci_path) if dag_ci_path else None
                else:
                    dag_plain_path = CausalTools._create_dag_plot(session_id, proposed_dag)
                    dag_path = Path(dag_plain_path) if dag_plain_path else None
            except Exception:
                dag_path = None
            add_plot(dag_path, "Figure 2: Directed Acyclic Graph (DAG) structure.", width_in=5.2, height_in=4.2)
        else:
            story.append(Paragraph("No DAG available.", styles["Normal"]))
        story.append(Spacer(1, 0.12 * inch))

        # 4) Methodology (tram-style narrative)
        story.append(Paragraph("4. Methodology", styles["Heading2"]))
        story.append(Paragraph("4.1 TRAM-DAG Model", styles["Heading3"]))
        story.append(
            Paragraph(
                "TRAM-DAG (Transformation Models for Directed Acyclic Graphs) combines transformation models with DAG-based causal structure learning. "
                "It estimates conditional distributions given parent variables and supports observational, interventional, and counterfactual-style reasoning.",
                styles["Normal"],
            )
        )
        story.append(Paragraph("4.2 DAG Consistency Testing", styles["Heading3"]))
        story.append(
            Paragraph(
                "Conditional independence (CI) tests compare model-implied independencies to empirical data. "
                "Rejected CI statements indicate potential misspecification in the current DAG.",
                styles["Normal"],
            )
        )
        story.append(Paragraph("4.3 Training Configuration", styles["Heading3"]))
        fit_info = session.get("fitted_model", {}) or {}
        seed = fit_info.get("random_seed", 42)
        story.append(Paragraph(f"<b>Random seed:</b> {seed}", styles["Normal"]))
        story.append(Spacer(1, 0.08 * inch))

        # 5) Model fitting & diagnostics
        story.append(Paragraph("5. Model Fitting & Diagnostics", styles["Heading2"]))
        story.append(Paragraph("5.1 DAG Consistency Tests", styles["Heading3"]))
        if ci_results:
            story.append(Paragraph(f"<b>Consistent with data:</b> {'Yes' if ci_results.get('consistent') else 'No'}", styles["Normal"]))
            story.append(Paragraph(f"<b>Rejected CI tests:</b> {ci_results.get('rejected_count', 0)}", styles["Normal"]))
            tests = ci_results.get("tests", []) or []
            if tests:
                shown = tests[:12]
                detail_lines = []
                for t in shown:
                    ci_str = t.get("ci", "CI statement")
                    p_val = t.get("adj_p_value", t.get("p_value", None))
                    rejected = bool(t.get("rejected", False))
                    p_txt = f"{float(p_val):.4g}" if isinstance(p_val, (int, float)) else "NA"
                    status = "REJECTED" if rejected else "PASSED"
                    detail_lines.append(f"• {ci_str} | p={p_txt} | {status}")
                story.append(Paragraph("<br/>".join(detail_lines), styles["Normal"]))
                if len(tests) > len(shown):
                    story.append(Paragraph(f"... and {len(tests) - len(shown)} more CI tests.", styles["Normal"]))
        else:
            story.append(Paragraph("No consistency tests performed yet.", styles["Normal"]))
        story.append(Spacer(1, 0.12 * inch))

        story.append(Paragraph("5.2 Fit Diagnostics Plots", styles["Heading3"]))
        fig_counter = 3  # Figures 1-2 are pairplot + DAG
        experiment_dir = session.get("experiment_dir")
        mcp_urls = session.get("mcp_plot_urls_by_kind", {}) or {}
        if experiment_dir and Path(experiment_dir).exists():
            story.append(Paragraph(f"<b>Experiment directory:</b> {experiment_dir}", styles["Normal"]))

            ordered_kinds = [
                ("5.2.1 Training & Validation Loss", "loss_history", "Training and validation loss history."),
                ("5.2.2 Linear Shift Parameters", "linear_shift_history", "Linear shift parameter history over training epochs."),
                ("5.2.3 Simple Intercepts", "simple_intercepts_history", "Simple intercept parameter history."),
                ("5.2.4 h-DAG Transformation Functions", "hdag", "h-DAG transformation functions."),
                ("5.2.5 Latent Distributions", "latents", "Latent distributions (should approach N(0,1))."),
                ("5.2.6 Sampling Distributions", "sampling_distributions", "Sampling distributions from fitted model."),
                ("5.2.7 Observational Samples vs True", "samples_vs_true", "Observational samples (orange) vs held-out test data (blue)."),
            ]
            for subsection, kind, desc in ordered_kinds:
                story.append(Paragraph(subsection, styles["Heading4"]))
                p = CausalTools._api_plot_url_to_path(mcp_urls.get(kind))
                if add_plot(p, f"Figure {fig_counter}: {desc}"):
                    fig_counter += 1

            # Fallback essentials if missing MCP artifacts.
            if "loss_history" not in mcp_urls:
                loss_p = CausalTools._create_loss_plot(experiment_dir, session_id=session_id)
                if add_plot(Path(loss_p) if loss_p else None, f"Figure {fig_counter}: Training and validation loss history."):
                    fig_counter += 1
            if "sampling_distributions" not in mcp_urls and "samples_vs_true" not in mcp_urls:
                dist_p = CausalTools._create_distribution_plot(experiment_dir, session_id, n_samples=10000)
                if add_plot(Path(dist_p) if dist_p else None, f"Figure {fig_counter}: Sampling distributions from fitted model."):
                    fig_counter += 1
        else:
            story.append(Paragraph("Model not fitted yet, so fit diagnostics are unavailable.", styles["Normal"]))
        story.append(Spacer(1, 0.12 * inch))

        # 6) Observational sampling
        story.append(Paragraph("6. Observational Sampling", styles["Heading2"]))
        story.append(
            Paragraph(
                "The model samples from the learned joint distribution without interventions. "
                "When fit quality is good, sampled distributions should align with held-out data.",
                styles["Normal"],
            )
        )
        obs_path = CausalTools._api_plot_url_to_path(mcp_urls.get("samples_vs_true"))
        if add_plot(obs_path, f"Figure {fig_counter}: Observational samples (orange) vs held-out test data (blue)."):
            fig_counter += 1
        story.append(Spacer(1, 0.08 * inch))

        # 7) Causal effect analysis (ATE)
        story.append(Paragraph("7. Causal Effect Analysis", styles["Heading2"]))
        story.append(Paragraph("7.1 Average Treatment Effect (ATE)", styles["Heading3"]))
        query_results = session.get("query_results", {}) or {}
        ate_data = query_results.get("ate")
        if ate_data:
            X = ate_data.get("X", "X")
            Y = ate_data.get("Y", "Y")
            story.append(Paragraph(f"<b>Treatment:</b> do({X} = {ate_data.get('x_treated', 'N/A')})", styles["Normal"]))
            story.append(Paragraph(f"<b>Control:</b> do({X} = {ate_data.get('x_control', 'N/A')})", styles["Normal"]))
            story.append(Paragraph(f"<b>Outcome:</b> {Y}", styles["Normal"]))
            story.append(Paragraph(f"<b>ATE:</b> {float(ate_data.get('ate', 0.0) or 0.0):.4f}", styles["Normal"]))
            story.append(Paragraph(
                f"E[{Y}|treated]={float(ate_data.get('y_treated_mean', 0.0) or 0.0):.4f}, "
                f"E[{Y}|control]={float(ate_data.get('y_control_mean', 0.0) or 0.0):.4f}",
                styles["Normal"],
            ))

        else:
            story.append(Paragraph("No ATE/interventional result available yet.", styles["Normal"]))
        story.append(Spacer(1, 0.12 * inch))

        # 8) Interventional sampling section title (tram wording)
        story.append(Paragraph("8. Interventional Sampling: Treated vs Control", styles["Heading2"]))
        story.append(
            Paragraph(
                "The plot compares treated and control interventions, and the distributional shift visualizes the estimated ATE.",
                styles["Normal"],
            )
        )
        inter_p = CausalTools._api_plot_url_to_path(mcp_urls.get("intervention"))
        if not inter_p and ate_data and experiment_dir and Path(experiment_dir).exists():
            inter_gen = CausalTools._create_intervention_plot(experiment_dir, ate_data)
            inter_p = Path(inter_gen) if inter_gen else None
        if add_plot(inter_p, f"Figure {fig_counter}: Treated (orange) vs control (blue) interventional distributions."):
            fig_counter += 1
        story.append(Spacer(1, 0.08 * inch))

        # Appendix: reproducibility
        story.append(PageBreak())
        story.append(Paragraph("Appendix: Reproducibility", styles["Heading2"]))
        story.append(
            Paragraph(
                "Reproducible scripts are exported to the experiment folders and are referenced below.",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 0.1 * inch))

        # A.1 Reproducibility exports
        exported = CausalTools._export_reproducibility_artifacts(session_id, session)
        story.append(Paragraph("A.1 Reproducibility Exports", styles["Heading3"]))
        if experiment_dir:
            story.append(Paragraph(f"<b>Experiment root:</b> {experiment_dir}", styles["Normal"]))
        if exported.get("scripts_dir"):
            story.append(Paragraph(f"<b>Scripts folder:</b> {exported.get('scripts_dir')}", styles["Normal"]))
        if exported.get("reproducible_package_dir"):
            story.append(Paragraph(f"<b>Reproducible package:</b> {exported.get('reproducible_package_dir')}", styles["Normal"]))
        story.append(Spacer(1, 0.1 * inch))

        # A.2 Sources and artifacts
        story.append(Paragraph("A.2 Sources and Artifacts", styles["Heading3"]))
        story.append(Paragraph(f"<b>Data file:</b> {session.get('data_path', 'N/A')}", styles["Normal"]))
        if session.get("data_info"):
            story.append(Paragraph(f"<b>Data schema:</b> {session.get('data_info')}", styles["Normal"]))
        if session.get("fitted_model"):
            fit_model = session.get("fitted_model", {}) or {}
            split_paths = fit_model.get("split_paths", {}) or {}
            if split_paths:
                story.append(Paragraph(f"<b>Train/val/test splits:</b> {split_paths}", styles["Normal"]))
            manifest = fit_model.get("artifact_manifest", []) or []
            if manifest:
                story.append(Paragraph(f"<b>Fit artifact manifest entries:</b> {len(manifest)}", styles["Normal"]))
        if mcp_urls:
            lines = [f"• {k}: {v}" for k, v in mcp_urls.items()]
            story.append(Paragraph("<b>Stored plot sources (kind → URL):</b><br/>" + "<br/>".join(lines), styles["Normal"]))

        # References (tram-like)
        story.append(Spacer(1, 0.12 * inch))
        story.append(Paragraph("References", styles["Heading2"]))
        refs = [
            "Sick, B., &amp; D&uuml;rr, O. (2025). Interpretable Neural Causal Models with TRAM-DAGs. <i>arXiv:2503.16206</i>.",
            "Hothorn, T., Most, L., &amp; B&uuml;hlmann, P. (2018). Most Likely Transformations. <i>Scandinavian Journal of Statistics</i>.",
            "Pearl, J. (2009). <i>Causality</i> (2nd ed.). Cambridge University Press.",
            "Hothorn, T., Kneib, T., &amp; B&uuml;hlmann, P. (2014). Conditional Transformation Models. <i>JRSS-B</i>.",
        ]
        for ref in refs:
            story.append(Paragraph(f"&bull; {ref}", styles["Normal"]))

        # Build PDF
        doc.build(story)

        # Clean session-scoped temp files only.
        if TEMP_PLOTS_DIR.exists():
            for f in TEMP_PLOTS_DIR.glob(f"{session_id}_*.png"):
                try:
                    f.unlink()
                except Exception:
                    pass

        return output_path
    
    @staticmethod
    def _compute_dag_layout(G, vars_list: List[str]) -> Dict:
        """
        Compute a hierarchical (top-to-bottom) layout for a DAG.
        
        - Multi-child layers: children are spread horizontally (tree-like)
        - Pure chains (each layer has 1 node): nodes are placed in an inverted
          triangle / V-shape so that every node is at a unique (x, y), and
          non-adjacent nodes are far apart — leaving clear space for CI arcs.
        
        For graphs that aren't valid DAGs (cycles), falls back to circular layout.
        """
        try:
            topo = list(nx.topological_sort(G))
        except nx.NetworkXError:
            return nx.circular_layout(G, scale=1.5)
        
        if len(topo) == 0:
            return {}
        
        # Assign each node to a layer = longest path from any root to that node
        layers = {n: 0 for n in G.nodes}
        for node in topo:
            for successor in G.successors(node):
                layers[successor] = max(layers[successor], layers[node] + 1)
        
        # Group nodes by layer
        layer_groups: Dict[int, List[str]] = {}
        for node, layer in layers.items():
            layer_groups.setdefault(layer, []).append(node)
        
        # Sort nodes within each layer for deterministic ordering
        for layer in layer_groups:
            layer_groups[layer].sort()
        
        max_layer = max(layers.values()) if layers else 0
        n_layers = max_layer + 1
        
        # Check if every layer has exactly 1 node (pure chain)
        is_chain = all(len(nodes) == 1 for nodes in layer_groups.values())
        
        pos = {}
        y_spacing = 2.0
        x_spacing = 2.5
        
        if is_chain and n_layers >= 3:
            # Pure chain with 3+ nodes — arrange as an inverted triangle:
            #
            #     x1 (top-center)
            #    /
            #  x2 (bottom-left)       x3 (bottom-right)
            #
            # Root at top-center, then spread children left→right at the
            # bottom.  This gives every pair a different geometric angle,
            # so CI arcs never overlap the causal edges.
            #
            # More precisely: distribute the chain nodes along a V-shape
            # where x fans out as layer increases.
            for layer, nodes in layer_groups.items():
                node = nodes[0]
                # Fan outward: each successive layer moves further from center
                # and alternates left/right, but each x is unique
                if layer == 0:
                    x = 0.0
                    y = 0.0
                else:
                    # Place nodes in an expanding fan
                    # Odd layers go left, even layers go right (except layer 0)
                    sign = -1 if layer % 2 == 1 else 1
                    x = sign * layer * 1.2
                    y = -layer * y_spacing
                pos[node] = np.array([x, y])
        else:
            # General case: tree layout with horizontal spreading per layer
            for layer, nodes in layer_groups.items():
                n_nodes = len(nodes)
                for i, node in enumerate(nodes):
                    if n_nodes > 1:
                        x = (i - (n_nodes - 1) / 2.0) * x_spacing
                    else:
                        x = 0.0
                    y = -layer * y_spacing
                    pos[node] = np.array([x, y])
        
        return pos
    
    @staticmethod
    def _create_dag_plot(session_id: str, proposed_dag: Dict) -> Optional[str]:
        """Create DAG visualization plot"""
        try:
            vars = proposed_dag.get('variables', [])
            edges = proposed_dag.get('edges', [])
            
            if not vars:
                return None
            
            G = nx.DiGraph()
            G.add_nodes_from(vars)
            G.add_edges_from(edges)
            
            # Hierarchical layout: parents on top, children below
            pos = CausalTools._compute_dag_layout(G, vars)
            
            # Get topological order for numbering
            try:
                topo_order = list(nx.topological_sort(G))
            except nx.NetworkXError:
                # If graph has cycles, use original order
                topo_order = vars
            
            # Create node number mapping (1-indexed)
            # Ensure all nodes are numbered, even if not in topo_order
            node_numbers = {}
            for i, node in enumerate(topo_order):
                node_numbers[node] = str(i+1)
            # Handle any nodes not in topo_order (shouldn't happen, but safety check)
            for node in vars:
                if node not in node_numbers:
                    node_numbers[node] = str(len(node_numbers) + 1)
            
            # Node colors based on parent/child relationships
            # Parents: nodes with outgoing edges (have children)
            # Children: nodes with incoming edges (have parents)
            parents = {n for n in G.nodes if G.out_degree(n) > 0}
            children = {n for n in G.nodes if G.in_degree(n) > 0}
            both = parents & children  # Nodes that are both parents and children
            
            node_colors = [
                "lightgreen" if n in both
                else "lightblue" if n in parents
                else "lightcoral" if n in children
                else "lightgray"  # Isolated nodes (shouldn't happen in DAG)
                for n in G.nodes
            ]
            
            # Create labels with numbers
            node_labels = {node: f"{node}\n({node_numbers[node]})" for node in G.nodes}
            
            plt.figure(figsize=(8, 6))
            nx.draw(G, pos, labels=node_labels, node_color=node_colors,
                   node_size=2000, font_size=9, font_weight='bold',
                   arrows=True, arrowsize=20, edge_color='gray', width=2)
            
            # Legend with parent/child terminology
            legend_elements = [
                mpatches.Patch(facecolor='lightgreen', label='Parent & Child'),
                mpatches.Patch(facecolor='lightblue', label='Parent'),
                mpatches.Patch(facecolor='lightcoral', label='Child')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            print(f"[DEBUG] DAG plot legend set with Parent/Child terminology")
            
            plt.title("Proposed DAG Structure", fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Save (use absolute path constant)
            img_path = TEMP_PLOTS_DIR / f"{session_id}_dag.png"
            
            # Delete old plot file if it exists to force regeneration
            if img_path.exists():
                try:
                    img_path.unlink()
                    print(f"[DEBUG] Deleted old DAG plot file to force regeneration")
                except Exception as e:
                    print(f"[WARNING] Could not delete old DAG plot file: {e}")
            print(f"[DEBUG] Saving DAG plot to: {img_path}")
            print(f"[DEBUG] Absolute path: {img_path.resolve()}")
            print(f"[DEBUG] temp_dir exists: {TEMP_PLOTS_DIR.exists()}, is_dir: {TEMP_PLOTS_DIR.is_dir()}")
            print(f"[DEBUG] temp_dir is writable: {os.access(TEMP_PLOTS_DIR, os.W_OK) if TEMP_PLOTS_DIR.exists() else 'N/A'}")
            try:
                plt.savefig(str(img_path), dpi=150, bbox_inches='tight')
                plt.close()
                # Force filesystem sync to ensure file is written to disk
                if img_path.exists():
                    with open(img_path, 'rb') as f:
                        pass
                    os.sync()  # Force OS to write buffered data to disk
                print(f"[DEBUG] plt.savefig completed for DAG plot")
            except Exception as save_error:
                print(f"[ERROR] Failed to save DAG plot: {save_error}")
                import traceback
                traceback.print_exc()
                plt.close()
                return None
            
            # Verify file was created and wait for filesystem sync
            import time
            time.sleep(0.1)  # Small delay to ensure filesystem sync
            if img_path.exists():
                file_size = img_path.stat().st_size
                print(f"[DEBUG] DAG plot successfully saved: {img_path} (size: {file_size} bytes)")
                # Double-check file is actually readable
                try:
                    with open(img_path, 'rb') as f:
                        f.read(1)  # Try to read first byte
                    print(f"[DEBUG] DAG plot file is readable")
                except Exception as read_error:
                    print(f"[WARNING] DAG plot file exists but is not readable: {read_error}")
            else:
                print(f"[ERROR] DAG plot file was not created: {img_path}")
            
            return str(img_path)
        except Exception as e:
            print(f"[ERROR] Failed to create DAG plot: {e}")
            return None
    
    @staticmethod
    def _compute_implied_cis(dag: Dict) -> List[Dict]:
        """
        Compute ALL implied conditional independencies from a DAG, matching
        dagitty::impliedConditionalIndependencies (including marginal CIs).
        
        Algorithm: for every pair of non-adjacent nodes (X, Y), search for
        the smallest conditioning set Z (including empty set) such that
        X ⊥ Y | Z (d-separated).
        
        Search order (smallest Z first):
          0. Empty set {} (marginal independence, e.g. colliders)
          1. Pa(X)
          2. Pa(Y)
          3. Pa(X) ∪ Pa(Y)
          4. Exhaustive search over subsets of size 1, 2, ... 
        
        Returns a list of dicts:
          [{"x": str, "y": str, "conditioning_set": List[str]}, ...]
        """
        from itertools import combinations

        def is_d_separated_version_safe(graph: nx.DiGraph, x_node: str, y_node: str, z_set: set) -> bool:
            """
            NetworkX changed d-separation APIs across versions.
            Try available variants in order and return False only when none work.
            """
            X = {x_node}
            Y = {y_node}
            Z = set(z_set)

            # Newer shorthand on nx namespace (if available)
            if hasattr(nx, "d_separated"):
                try:
                    return bool(nx.d_separated(graph, X, Y, Z))
                except Exception:
                    pass

            # Version-dependent functions in algorithms.d_separation
            try:
                from networkx.algorithms.d_separation import d_separated as nx_d_separated  # type: ignore
                return bool(nx_d_separated(graph, X, Y, Z))
            except Exception:
                pass

            try:
                from networkx.algorithms.d_separation import is_d_separator  # type: ignore
                return bool(is_d_separator(graph, X, Y, Z))
            except Exception:
                pass

            return False
        
        vars_list = dag.get('variables', [])
        edges = dag.get('edges', [])
        
        if not vars_list or not edges:
            return []
        
        G = nx.DiGraph()
        G.add_nodes_from(vars_list)
        G.add_edges_from(edges)
        
        # Verify the graph is a DAG
        if not nx.is_directed_acyclic_graph(G):
            print("[WARNING] _compute_implied_cis: graph has cycles, returning empty")
            return []
        
        # Build set of adjacent pairs (edges in either direction)
        adj_pairs = set()
        for u, v in G.edges():
            adj_pairs.add((u, v))
            adj_pairs.add((v, u))
        
        implied_cis = []
        
        for i, x in enumerate(vars_list):
            for y in vars_list[i + 1:]:
                # Skip adjacent pairs — no CI to test
                if (x, y) in adj_pairs or (y, x) in adj_pairs:
                    continue
                
                other_vars = [v for v in vars_list if v != x and v != y]
                
                # --- 0. Check marginal independence (empty conditioning set) ---
                found_z = None
                if is_d_separated_version_safe(G, x, y, set()):
                    found_z = set()  # marginal independence
                
                # --- 1-3. Try smart non-empty candidates ---
                if found_z is None:
                    parents_x = set(G.predecessors(x))
                    parents_y = set(G.predecessors(y))
                    
                    candidates = []
                    if parents_x:
                        candidates.append(parents_x)
                    if parents_y:
                        candidates.append(parents_y)
                    union_parents = parents_x | parents_y
                    if union_parents and union_parents not in candidates:
                        candidates.append(union_parents)
                    
                    for z_set in candidates:
                        if is_d_separated_version_safe(G, x, y, z_set):
                            found_z = z_set
                            break
                    
                    # --- 4. Fallback: exhaustive search (subsets of increasing size) ---
                    if found_z is None and other_vars:
                        for size in range(1, len(other_vars) + 1):
                            for subset in combinations(other_vars, size):
                                z_set = set(subset)
                                if z_set in candidates:
                                    continue
                                if is_d_separated_version_safe(G, x, y, z_set):
                                    found_z = z_set
                                    break
                            if found_z is not None:
                                break
                
                if found_z is not None:
                    pair_key = tuple(sorted([x, y]))
                    implied_cis.append({
                        "x": pair_key[0],
                        "y": pair_key[1],
                        "conditioning_set": sorted(list(found_z))
                    })
        
        print(f"[DEBUG] _compute_implied_cis found {len(implied_cis)} CIs for "
              f"{len(vars_list)} vars, {len(edges)} edges")
        for ci in implied_cis:
            cond_str = ', '.join(ci['conditioning_set']) if ci['conditioning_set'] else '∅ (marginal)'
            print(f"[DEBUG]   {ci['x']} ⊥ {ci['y']} | {{{cond_str}}}")
        
        return implied_cis
    
    @staticmethod
    def _create_dag_plot_with_ci(session_id: str, proposed_dag: Dict) -> Optional[tuple]:
        """
        Create DAG visualization plot with implied CI tests shown as red dashed arcs.
        
        - Solid gray arrows: causal edges in the DAG
        - Red dashed arcs: pairs of variables that are implied to be conditionally 
          independent (d-separated) given some conditioning set, annotated with the 
          conditioning set.  Arcs curve away from the straight edges to avoid overlap.
        """
        try:
            vars = proposed_dag.get('variables', [])
            edges = proposed_dag.get('edges', [])
            
            if not vars:
                return None
            
            G = nx.DiGraph()
            G.add_nodes_from(vars)
            G.add_edges_from(edges)
            
            # Hierarchical layout: parents on top, children below.
            # CI arcs between non-adjacent nodes curve to the side,
            # clearly separated from the straight downward causal edges.
            pos = CausalTools._compute_dag_layout(G, vars)
            
            # Get topological order for numbering
            try:
                topo_order = list(nx.topological_sort(G))
            except nx.NetworkXError:
                topo_order = vars
            
            node_numbers = {}
            for i, node in enumerate(topo_order):
                node_numbers[node] = str(i + 1)
            for node in vars:
                if node not in node_numbers:
                    node_numbers[node] = str(len(node_numbers) + 1)
            
            # Node colors
            parents = {n for n in G.nodes if G.out_degree(n) > 0}
            children = {n for n in G.nodes if G.in_degree(n) > 0}
            both = parents & children
            
            node_colors = [
                "lightgreen" if n in both
                else "lightblue" if n in parents
                else "lightcoral" if n in children
                else "lightgray"
                for n in G.nodes
            ]
            
            node_labels = {node: f"{node}\n({node_numbers[node]})" for node in G.nodes}
            
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Draw the main DAG edges (straight arrows)
            nx.draw(G, pos, labels=node_labels, node_color=node_colors,
                    node_size=2000, font_size=9, font_weight='bold',
                    arrows=True, arrowsize=20, edge_color='gray', width=2,
                    ax=ax)
            
            # Compute implied CIs and draw them as red dashed arcs
            implied_cis = CausalTools._compute_implied_cis(proposed_dag)
            
            # Build set of existing edges for overlap detection
            edge_set = set()
            for u, v in edges:
                edge_set.add((u, v))
                edge_set.add((v, u))
            
            ci_annotations = []
            for idx, ci in enumerate(implied_cis):
                x, y = ci["x"], ci["y"]
                cond_set = ci["conditioning_set"]
                
                if x in pos and y in pos:
                    x_pos = pos[x]
                    y_pos = pos[y]
                    
                    # Use curved arcs so CI lines don't overlap with straight edges
                    ax.annotate(
                        "", xy=y_pos, xytext=x_pos,
                        arrowprops=dict(
                            arrowstyle="-",
                            color="#DC2626",
                            linestyle="dashed",
                            linewidth=1.5,
                            alpha=0.7
                        )
                    )
                    
                    # Place label at midpoint
                    mid_x = (x_pos[0] + y_pos[0]) / 2
                    mid_y = (x_pos[1] + y_pos[1]) / 2
                    perp_x = 0
                    perp_y = 0.06
                    
                    if cond_set:
                        ci_label = f"{x} \u27C2 {y} | {', '.join(cond_set)}"
                    else:
                        ci_label = f"{x} \u27C2 {y}"
                    
                    ax.text(mid_x + perp_x, mid_y + perp_y, ci_label,
                            fontsize=7, color="#DC2626", ha='center', va='center',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                      edgecolor='#DC2626', alpha=0.85))
                    
                    ci_annotations.append(ci_label)
            
            # Legend
            legend_elements = [
                mpatches.Patch(facecolor='lightgreen', label='Parent & Child'),
                mpatches.Patch(facecolor='lightblue', label='Parent'),
                mpatches.Patch(facecolor='lightcoral', label='Child'),
                plt.Line2D([0], [0], color='gray', linewidth=2, label='Causal Edge'),
                plt.Line2D([0], [0], color='#DC2626', linewidth=1.5, linestyle='dashed',
                           label=f'Implied CI Test ({len(implied_cis)})')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            title = "DAG Structure with Implied Conditional Independence Tests"
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.axis('off')
            fig.tight_layout()
            
            # Save
            img_path = TEMP_PLOTS_DIR / f"{session_id}_dag_ci.png"
            if img_path.exists():
                try:
                    img_path.unlink()
                except Exception:
                    pass
            
            plt.savefig(str(img_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Verify
            import time
            time.sleep(0.1)
            if img_path.exists():
                print(f"[DEBUG] DAG+CI plot saved: {img_path} ({img_path.stat().st_size} bytes)")
            else:
                print(f"[ERROR] DAG+CI plot not created: {img_path}")
                return None
            
            return str(img_path), implied_cis
        except Exception as e:
            print(f"[ERROR] Failed to create DAG+CI plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _create_dag_plot_with_ci_results(
        session_id: str,
        proposed_dag: Dict,
        ci_results: Dict
    ) -> Optional[str]:
        """
        Create DAG visualization plot with CI test results overlay.
        
        - Solid gray arrows: causal edges in the DAG
        - Green dashed arcs: CI tests that PASSED (not rejected)
        - Red dashed arcs: CI tests that were REJECTED
        - Arcs curve away from straight edges to avoid overlap
        - Bottom table: lists every CI statement with its p-value and pass/reject status
        
        Args:
            session_id: Session identifier for file naming
            proposed_dag: Dict with 'variables' and 'edges'
            ci_results: Dict from R bridge with 'tests' list containing individual test results
            
        Returns:
            Path to the saved plot image, or None on error
        """
        try:
            vars_list = proposed_dag.get('variables', [])
            edges = proposed_dag.get('edges', [])
            tests = ci_results.get('tests', [])
            
            if not vars_list:
                return None
            
            G = nx.DiGraph()
            G.add_nodes_from(vars_list)
            G.add_edges_from(edges)
            
            # Hierarchical layout: parents on top, children below
            pos = CausalTools._compute_dag_layout(G, vars_list)

            # Topological order for numbering
            try:
                topo_order = list(nx.topological_sort(G))
            except nx.NetworkXError:
                topo_order = vars_list
            
            node_numbers = {}
            for i, node in enumerate(topo_order):
                node_numbers[node] = str(i + 1)
            for node in vars_list:
                if node not in node_numbers:
                    node_numbers[node] = str(len(node_numbers) + 1)
            
            # Node colors
            parent_nodes = {n for n in G.nodes if G.out_degree(n) > 0}
            child_nodes = {n for n in G.nodes if G.in_degree(n) > 0}
            both_nodes = parent_nodes & child_nodes
            
            node_colors = [
                "lightgreen" if n in both_nodes
                else "lightblue" if n in parent_nodes
                else "lightcoral" if n in child_nodes
                else "lightgray"
                for n in G.nodes
            ]
            
            node_labels = {node: f"{node}\n({node_numbers[node]})" for node in G.nodes}
            
            # Parse CI test results to extract variable pairs
            # CI format from R: "X _||_ Y | Z1, Z2" or "X _||_ Y"
            parsed_tests = []
            for test in tests:
                ci_str = test.get('ci', '')
                rejected = test.get('rejected', False)
                p_value = test.get('adj_p_value', test.get('p_value', None))
                test_name = (
                    test.get('test_name')
                    or test.get('test')
                    or test.get('method')
                    or test.get('ci_test')
                    or test.get('test_type')
                    or ""
                )
                
                # Parse "X _||_ Y | Z1, Z2" format
                norm_ci = str(ci_str).replace('⟂', '_||_').replace('⊥', '_||_').replace('⫫', '_||_')
                parts = norm_ci.split('_||_')
                if len(parts) == 2:
                    x_var = parts[0].strip().strip('()[]')
                    rest = parts[1].strip().strip('()[]')
                    # Split conditioning set
                    if '|' in rest:
                        y_and_z = rest.split('|', 1)
                        y_var = y_and_z[0].strip().strip('()[]')
                        cond_set = y_and_z[1].strip()
                    else:
                        y_var = rest.strip().strip('()[]')
                        cond_set = ''
                    
                    parsed_tests.append({
                        'x': x_var,
                        'y': y_var,
                        'cond_set': cond_set,
                        'rejected': rejected,
                        'p_value': p_value,
                        'test_name': test_name,
                        'ci_str': ci_str
                    })
            
            # If no tests could be parsed (e.g., R bridge not available), 
            # fall back to implied CIs from d-separation
            if not parsed_tests and not tests:
                implied = CausalTools._compute_implied_cis(proposed_dag)
                for ci in implied:
                    cond = ', '.join(ci['conditioning_set']) if ci['conditioning_set'] else ''
                    parsed_tests.append({
                        'x': ci['x'],
                        'y': ci['y'],
                        'cond_set': cond,
                        'rejected': False,
                        'p_value': None,
                        'test_name': 'implied',
                        'ci_str': f"{ci['x']} _||_ {ci['y']}" + (f" | {cond}" if cond else "")
                    })
            
            # Count rejected vs passed
            n_rejected = sum(1 for t in parsed_tests if t['rejected'])
            n_passed = len(parsed_tests) - n_rejected
            
            # Determine figure height: base for graph + extra for the table at the bottom
            n_tests_display = len(parsed_tests)
            table_height = max(1.5, 0.35 * n_tests_display + 0.8)
            fig_height = 7 + table_height
            
            fig = plt.figure(figsize=(10, fig_height))
            
            # Create grid: top for graph, bottom for table
            gs = fig.add_gridspec(2, 1, height_ratios=[7, table_height], hspace=0.05)
            ax_graph = fig.add_subplot(gs[0])
            ax_table = fig.add_subplot(gs[1])
            
            # --- Draw the DAG graph (straight arrows) ---
            nx.draw(G, pos, labels=node_labels, node_color=node_colors,
                    node_size=2000, font_size=9, font_weight='bold',
                    arrows=True, arrowsize=20, edge_color='gray', width=2,
                    ax=ax_graph)
            
            # Draw CI test result arcs on the graph
            # Group tests by variable pair (take worst result for drawing)
            pair_results = {}
            for t in parsed_tests:
                pair_key = (min(t['x'], t['y']), max(t['x'], t['y']))
                if pair_key not in pair_results:
                    pair_results[pair_key] = t
                elif t['rejected'] and not pair_results[pair_key]['rejected']:
                    pair_results[pair_key] = t
            
            passed_edges: List[Tuple[str, str]] = []
            rejected_edges: List[Tuple[str, str]] = []
            pair_meta: List[Tuple[str, str, Dict[str, Any]]] = []
            for (x_var, y_var), t in pair_results.items():
                if x_var in pos and y_var in pos:
                    edge = (x_var, y_var)
                    if t['rejected']:
                        rejected_edges.append(edge)
                    else:
                        passed_edges.append(edge)
                    pair_meta.append((x_var, y_var, t))

            # Draw CI dashed lines with NetworkX to align exactly node-to-node.
            if passed_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=passed_edges, style="dashed", width=2.0,
                    edge_color="#059669", arrows=False, alpha=0.75, ax=ax_graph
                )
            if rejected_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=rejected_edges, style="dashed", width=2.0,
                    edge_color="#DC2626", arrows=False, alpha=0.8, ax=ax_graph
                )

            # Add compact status badges near line midpoints.
            for x_var, y_var, t in pair_meta:
                x_pos = pos[x_var]
                y_pos = pos[y_var]
                mid_x = (x_pos[0] + y_pos[0]) / 2.0
                mid_y = (x_pos[1] + y_pos[1]) / 2.0
                color = "#DC2626" if t['rejected'] else "#059669"
                status = "REJECTED" if t['rejected'] else "OK"
                ax_graph.text(
                    mid_x, mid_y + 0.06, status,
                    fontsize=8, color=color, ha='center', va='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, alpha=0.9)
                )
            
            # Graph legend
            legend_elements = [
                mpatches.Patch(facecolor='lightgreen', label='Parent & Child'),
                mpatches.Patch(facecolor='lightblue', label='Parent'),
                mpatches.Patch(facecolor='lightcoral', label='Child'),
                plt.Line2D([0], [0], color='gray', linewidth=2, label='Causal Edge'),
                plt.Line2D([0], [0], color='#059669', linewidth=2, linestyle='dashed',
                           label=f'CI Passed ({n_passed})'),
                plt.Line2D([0], [0], color='#DC2626', linewidth=2, linestyle='dashed',
                           label=f'CI Rejected ({n_rejected})')
            ]
            ax_graph.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            consistency = "CONSISTENT" if n_rejected == 0 else "INCONSISTENT"
            title_color = "#059669" if n_rejected == 0 else "#DC2626"
            ax_graph.set_title(
                f"DAG Consistency Test Results — {consistency}",
                fontsize=13, fontweight='bold', color=title_color
            )
            ax_graph.axis('off')
            
            # --- Draw the CI tests table at the bottom ---
            ax_table.axis('off')
            
            if parsed_tests:
                # Build table data
                col_labels = ['#', 'CI Statement', 'Test', 'Adj. p-value', 'Result']
                table_data = []
                for i, t in enumerate(parsed_tests, 1):
                    ci_display = f"{t['x']} \u27C2 {t['y']}"
                    if t['cond_set']:
                        ci_display += f" | {t['cond_set']}"
                    
                    p_str = f"{t['p_value']:.4f}" if t['p_value'] is not None else "N/A"
                    result_str = "REJECTED" if t['rejected'] else "Passed"
                    test_raw = str(t['test_name']).strip() if t['test_name'] is not None else ""
                    test_str = test_raw.upper() if test_raw else "CI"
                    
                    table_data.append([str(i), ci_display, test_str, p_str, result_str])
                
                table = ax_table.table(
                    cellText=table_data,
                    colLabels=col_labels,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.05, 0.42, 0.13, 0.18, 0.15]
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.3)
                
                # Style the table
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        # Header row
                        cell.set_facecolor('#2C5F7D')
                        cell.set_text_props(color='white', fontweight='bold')
                    else:
                        # Data rows
                        test_idx = row - 1
                        if test_idx < len(parsed_tests):
                            if parsed_tests[test_idx]['rejected']:
                                cell.set_facecolor('#FEE2E2')  # Light red
                                if col == 4:  # Result column
                                    cell.set_text_props(color='#DC2626', fontweight='bold')
                            else:
                                cell.set_facecolor('#ECFDF5')  # Light green
                                if col == 4:
                                    cell.set_text_props(color='#059669', fontweight='bold')
                    cell.set_edgecolor('#E5E7EB')
                
                ax_table.set_title(
                    "Conditional Independence Test Details",
                    fontsize=11, fontweight='bold', color='#2C5F7D',
                    pad=10
                )
            else:
                ax_table.text(0.5, 0.5, "No CI test details available",
                              ha='center', va='center', fontsize=10, color='#6B7280')
            
            fig.tight_layout()
            
            # Save
            img_path = TEMP_PLOTS_DIR / f"{session_id}_dag_ci_results.png"
            if img_path.exists():
                try:
                    img_path.unlink()
                except Exception:
                    pass
            
            plt.savefig(str(img_path), dpi=150, bbox_inches='tight')
            plt.close()
            
            import time
            time.sleep(0.1)
            if img_path.exists():
                print(f"[DEBUG] DAG+CI results plot saved: {img_path} ({img_path.stat().st_size} bytes)")
                return str(img_path)
            else:
                print(f"[ERROR] DAG+CI results plot not created: {img_path}")
                return None
        except Exception as e:
            print(f"[ERROR] Failed to create DAG+CI results plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _create_loss_plot(experiment_dir: str, session_id: Optional[str] = None) -> Optional[str]:
        """Create training loss history plot"""
        try:
            experiment_path = Path(experiment_dir)
            if not experiment_path.exists():
                return None
            
            # Find all variable directories
            var_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            
            loss_data = {}
            for var_dir in var_dirs:
                var = var_dir.name
                train_file = var_dir / "train_loss_hist.json"
                val_file = var_dir / "val_loss_hist.json"
                
                if train_file.exists():
                    try:
                        with open(train_file) as f:
                            train_loss = json.load(f)
                            if isinstance(train_loss, list):
                                loss_data[f"{var}_train"] = train_loss
                    except:
                        pass
                
                if val_file.exists():
                    try:
                        with open(val_file) as f:
                            val_loss = json.load(f)
                            if isinstance(val_loss, list):
                                loss_data[f"{var}_val"] = val_loss
                    except:
                        pass
            
            if not loss_data:
                return None
            
            plt.figure(figsize=(10, 6))
            colors_list = plt.cm.tab10(range(len(var_dirs)))
            
            for i, var_dir in enumerate(var_dirs):
                var = var_dir.name
                train_key = f"{var}_train"
                val_key = f"{var}_val"
                
                if train_key in loss_data:
                    epochs = range(1, len(loss_data[train_key]) + 1)
                    plt.plot(epochs, loss_data[train_key], 
                           label=f"{var} (train)", color=colors_list[i], linestyle='-', linewidth=2)
                
                if val_key in loss_data:
                    epochs = range(1, len(loss_data[val_key]) + 1)
                    plt.plot(epochs, loss_data[val_key],
                           label=f"{var} (val)", color=colors_list[i], linestyle='--', linewidth=2)
            
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss (NLL)', fontsize=12)
            plt.title('Training and Validation Loss History', fontsize=14, fontweight='bold')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save with session_id if provided, otherwise use experiment_path.name (use absolute path constant)
            if session_id:
                img_path = TEMP_PLOTS_DIR / f"{session_id}_loss.png"
            else:
                img_path = TEMP_PLOTS_DIR / f"{experiment_path.name}_loss.png"
            print(f"[DEBUG] Saving loss plot to: {img_path}")
            print(f"[DEBUG] Absolute path: {img_path.resolve()}")
            print(f"[DEBUG] temp_dir exists: {TEMP_PLOTS_DIR.exists()}, is_dir: {TEMP_PLOTS_DIR.is_dir()}")
            print(f"[DEBUG] temp_dir is writable: {os.access(TEMP_PLOTS_DIR, os.W_OK) if TEMP_PLOTS_DIR.exists() else 'N/A'}")
            try:
                plt.savefig(str(img_path), dpi=150, bbox_inches='tight')
                plt.close()
                # Force filesystem sync to ensure file is written to disk
                if img_path.exists():
                    # Open and close file to force flush
                    with open(img_path, 'rb') as f:
                        pass
                    os.sync()  # Force OS to write buffered data to disk
                print(f"[DEBUG] plt.savefig completed for loss plot")
            except Exception as save_error:
                print(f"[ERROR] Failed to save loss plot: {save_error}")
                import traceback
                traceback.print_exc()
                plt.close()
                return None
            
            # Verify file was created and wait a moment for filesystem sync
            import time
            time.sleep(0.1)  # Small delay to ensure filesystem sync
            if img_path.exists():
                file_size = img_path.stat().st_size
                print(f"[DEBUG] Loss plot successfully saved: {img_path} (size: {file_size} bytes)")
                # Double-check file is actually readable
                try:
                    with open(img_path, 'rb') as f:
                        f.read(1)  # Try to read first byte
                    print(f"[DEBUG] Loss plot file is readable")
                except Exception as read_error:
                    print(f"[WARNING] Loss plot file exists but is not readable: {read_error}")
            else:
                print(f"[ERROR] Loss plot file was not created: {img_path}")
            
            return str(img_path)
        except Exception as e:
            print(f"[ERROR] Failed to create loss plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _create_distribution_plot(experiment_dir: str, session_id: str, n_samples: int = 10000) -> Optional[str]:
        """Create sampling distribution plot"""
        try:
            # Sample from model
            samples = CausalTools.sample_from_model(experiment_dir, n_samples=n_samples)
            
            if not samples:
                return None
            
            # Create subplots
            n_vars = len(samples)
            fig, axes = plt.subplots(1, n_vars, figsize=(4*n_vars, 4))
            if n_vars == 1:
                axes = [axes]
            
            for i, (var, data) in enumerate(samples.items()):
                values = data['values']
                axes[i].hist(values, bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{var} Distribution', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Value', fontsize=10)
                axes[i].set_ylabel('Frequency', fontsize=10)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save (use absolute path constant)
            img_path = TEMP_PLOTS_DIR / f"{session_id}_distributions.png"
            print(f"[DEBUG] Saving distribution plot to: {img_path}")
            print(f"[DEBUG] Absolute path: {img_path.resolve()}")
            print(f"[DEBUG] temp_dir exists: {TEMP_PLOTS_DIR.exists()}, is_dir: {TEMP_PLOTS_DIR.is_dir()}")
            print(f"[DEBUG] temp_dir is writable: {os.access(TEMP_PLOTS_DIR, os.W_OK) if TEMP_PLOTS_DIR.exists() else 'N/A'}")
            try:
                plt.savefig(str(img_path), dpi=150, bbox_inches='tight')
                plt.close()
                # Force filesystem sync to ensure file is written to disk
                if img_path.exists():
                    with open(img_path, 'rb') as f:
                        pass
                    os.sync()  # Force OS to write buffered data to disk
                print(f"[DEBUG] plt.savefig completed for distribution plot")
            except Exception as save_error:
                print(f"[ERROR] Failed to save distribution plot: {save_error}")
                import traceback
                traceback.print_exc()
                plt.close()
                return None
            
            # Verify file was created and wait for filesystem sync
            import time
            time.sleep(0.1)  # Small delay to ensure filesystem sync
            if img_path.exists():
                file_size = img_path.stat().st_size
                print(f"[DEBUG] Distribution plot successfully saved: {img_path} (size: {file_size} bytes)")
                # Double-check file is actually readable
                try:
                    with open(img_path, 'rb') as f:
                        f.read(1)  # Try to read first byte
                    print(f"[DEBUG] Distribution plot file is readable")
                except Exception as read_error:
                    print(f"[WARNING] Distribution plot file exists but is not readable: {read_error}")
            else:
                print(f"[ERROR] Distribution plot file was not created: {img_path}")
            
            return str(img_path)
        except Exception as e:
            print(f"[ERROR] Failed to create distribution plot: {e}")
            return None
    
    @staticmethod
    def _create_intervention_plot(experiment_dir: str, ate_data: Dict) -> Optional[str]:
        """Create intervention comparison plot"""
        try:
            X = ate_data.get('X')
            Y = ate_data.get('Y')
            x_treated = ate_data.get('x_treated', 1.0)
            x_control = ate_data.get('x_control', 0.0)
            
            # Sample under treatment and control
            treated_samples = CausalTools.sample_from_model(
                experiment_dir, n_samples=10000,
                do_interventions={X: x_treated}
            )
            control_samples = CausalTools.sample_from_model(
                experiment_dir, n_samples=10000,
                do_interventions={X: x_control}
            )
            
            if Y not in treated_samples or Y not in control_samples:
                return None
            
            y_treated = treated_samples[Y]['values']
            y_control = control_samples[Y]['values']
            
            plt.figure(figsize=(10, 6))
            
            plt.hist(y_control, bins=50, alpha=0.6, label=f'Control (do({X}={x_control}))',
                    color='blue', edgecolor='black')
            plt.hist(y_treated, bins=50, alpha=0.6, label=f'Treated (do({X}={x_treated}))',
                    color='red', edgecolor='black')
            
            plt.axvline(np.mean(y_control), color='blue', linestyle='--', linewidth=2,
                       label=f'Control Mean: {np.mean(y_control):.4f}')
            plt.axvline(np.mean(y_treated), color='red', linestyle='--', linewidth=2,
                       label=f'Treated Mean: {np.mean(y_treated):.4f}')
            
            plt.xlabel(f'{Y} Value', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Intervention Effect: {X} on {Y}', fontsize=14, fontweight='bold')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save
            img_path = TEMP_PLOTS_DIR / f"{Path(experiment_dir).name}_intervention.png"
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return str(img_path)
        except Exception as e:
            print(f"[ERROR] Failed to create intervention plot: {e}")
            return None

    @staticmethod
    def _create_data_pairplot(session_id: str, df: pd.DataFrame) -> Optional[str]:
        """Create upload-time pairplot for the uploaded dataset."""
        try:
            if df is None or df.empty:
                return None

            numeric_df = df.select_dtypes(include=[np.number]).copy()
            if numeric_df.shape[1] == 0:
                return None

            # Keep plotting responsive.
            if len(numeric_df) > 2500:
                numeric_df = numeric_df.sample(n=2500, random_state=42)
            if numeric_df.shape[1] > 8:
                numeric_df = numeric_df.iloc[:, :8]

            from pandas.plotting import scatter_matrix

            plt.close("all")
            axes = scatter_matrix(
                numeric_df,
                alpha=0.3,
                figsize=(11, 11),
                diagonal="hist",
                color="#4A90A4",
                hist_kwds={"bins": 30, "alpha": 0.75},
            )
            for ax in np.array(axes).ravel():
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
                ax.tick_params(axis="both", labelsize=7)

            plt.suptitle("Uploaded Data Pairplot", fontsize=12, y=1.02)
            plt.tight_layout()

            img_path = TEMP_PLOTS_DIR / f"{session_id}_data_pairplot.png"
            plt.savefig(str(img_path), dpi=130, bbox_inches="tight")
            plt.close("all")
            return f"/api/plots/{img_path.name}"
        except Exception as e:
            print(f"[WARNING] Could not create upload pairplot: {e}")
            try:
                plt.close("all")
            except Exception:
                pass
            return None


# ============================================================================
# AI AGENT — Receives user messages, selects tools, returns responses
# ============================================================================
#
#   User Message → AgentOrchestrator.process_message()
#                    → _agent_select_and_execute_tool()
#                        → LLM picks tool via function calling
#                        → _execute_tool() runs it locally
#                        → _format_tool_results() builds response
#                    → Response sent to Chat UI
#
# ============================================================================

class AgentOrchestrator:
    """
    Orchestrates the causal inference workflow using MCP (Model Context Protocol)
    
    Architecture:
    - Chat UI provides natural language input
    - LLM uses function calling to select tools from ToolRegistry
    - Tools execute locally via CausalTools (TRAM-DAG, R CI tests, etc.)
    - Results are formatted and returned to the user via chat UI
    
    Primary mode: AI Agent (LLM function calling → tool execution)
    Fallback mode: Workflow-based routing (if agent fails)
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.tools = CausalTools()  # Tool implementations
        
        # Initialize state - restore from session if it exists
        session = sessions.get(session_id, {})
        restored_step = session.get("current_step", "initial")
        print(f"[DEBUG] AgentOrchestrator init: Restoring current_step from session: {restored_step}")
        self.state = SessionState(
            session_id=session_id,
            messages=session.get("messages", []),
            current_step=restored_step
        )
        
        # Load data from session if available
        if session.get("data_path"):
            self.state.data_path = session["data_path"]
        if session.get("data_df"):
            try:
                # Reconstruct DataFrame from stored dict
                self.state.data_df = pd.DataFrame(session["data_df"])
                print(f"[DEBUG] Loaded DataFrame with shape {self.state.data_df.shape} and columns {list(self.state.data_df.columns)}")
            except Exception as e:
                print(f"[ERROR] Failed to reconstruct DataFrame: {e}")
                self.state.data_df = None
        if session.get("proposed_dag"):
            self.state.proposed_dag = session["proposed_dag"]
        if session.get("ci_test_results"):
            self.state.ci_test_results = session["ci_test_results"]
        if session.get("fitted_model"):
            self.state.fitted_model = session["fitted_model"]
        if session.get("experiment_dir"):
            self.state.experiment_dir = session["experiment_dir"]
        if session.get("pending_tool"):
            self.state.pending_tool = session["pending_tool"]
        if session.get("pending_tool_args"):
            self.state.pending_tool_args = session["pending_tool_args"]
        if session.get("pending_missing_param"):
            self.state.pending_missing_param = session["pending_missing_param"]
    
    def _clean_llm_response(self, text: str) -> str:
        """
        Clean up LLM responses for better formatting:
        - Remove LaTeX math notation (convert \\(x\\) to x)
        - Remove excessive line breaks
        - Clean up spacing
        - Format DAG structures better
        """
        if not text:
            return text
        
        # Remove LaTeX math notation: \(...\) and \[...\]
        text = re.sub(r'\\?\(([^)]+)\)', r'\1', text)  # \(x1\) -> x1
        text = re.sub(r'\\?\[([^\]]+)\]', r'\1', text)  # \[x1\] -> x1
        
        # Remove markdown code blocks that are just variable names
        text = re.sub(r'`([a-zA-Z0-9_]+)`', r'\1', text)  # `x1` -> x1 (but keep code blocks)
        
        # Clean up DAG structure formatting
        text = re.sub(r'(\w+)\s*→\s*(\w+)', r'\1 → \2', text)  # Normalize arrows
        text = re.sub(r'(\w+)\s*->\s*(\w+)', r'\1 → \2', text)  # Convert -> to →
        
        # Remove excessive line breaks (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Clean up spacing around special characters
        text = re.sub(r'\s+→\s+', ' → ', text)
        text = re.sub(r'\s+-\s+', ' - ', text)
        
        # Remove empty lines at start/end
        text = text.strip()
        
        # Format section headers better
        text = re.sub(r'^###\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
        text = re.sub(r'^##\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
        
        return text
    
    async def _generate_medical_explanation(self, result_type: str, result_data: Dict, context: str = "") -> str:
        """
        Generate a simple, medical-friendly explanation of statistical results using LLM.
        
        Args:
            result_type: Type of result ('intervention', 'ate', 'association', 'plot')
            result_data: Dictionary containing the result data
            context: Additional context about the analysis
        
        Returns:
            Simple explanation text suitable for medical doctors
        """
        try:
            # Build context about the analysis
            dag_vars = self.state.proposed_dag.get("variables", []) if self.state.proposed_dag else []
            var_context = f"Available variables: {', '.join(dag_vars)}" if dag_vars else ""
            
            sys_msg = """You are a medical statistics expert explaining causal analysis results to practicing physicians.
Your explanations should be:
- Concise (2-4 short paragraphs, maximum 150 words total)
- Simple and clear (avoid complex statistical jargon)
- Structured with clear paragraph breaks
- Medical context-aware (use medical analogies when helpful)
- Actionable (explain what the results mean for clinical practice)
- Honest about limitations (mention uncertainty when appropriate)

Write in a friendly, professional, scientific tone. Use 2-4 short, focused paragraphs separated by blank lines.
Each paragraph should cover one key point. Keep explanations brief and to the point.
DO NOT use markdown headers (###, ##, #), numbered lists (1., 2., 3.), or bullet points.
Write in clear, structured prose that is easy to scan and understand."""
            
            if result_type == "intervention":
                var = result_data.get("variable", "variable")
                value = result_data.get("intervention_value", 0)
                samples = result_data.get("samples", {})
                
                usr_msg = f"""A physician asked: "What if {var} = {value}?"

The analysis performed an intervention (like a clinical trial where we force {var} to be {value}).

Results:
"""
                for node, data in samples.items():
                    mean_val = data.get("mean", 0)
                    std_val = data.get("std", 0)
                    usr_msg += f"- {node}: mean = {mean_val:.4f}, standard deviation = {std_val:.4f}\n"
                
                usr_msg += f"""
{var_context}
{context}

Explain what these results mean in simple terms for a medical doctor. Write 2-4 short, focused paragraphs (maximum 150 words total) separated by blank lines. Each paragraph should cover one key point.

First paragraph: Briefly explain what the intervention means and the key finding (the mean values).
Second paragraph: Explain what the standard deviations tell us about variability and individual differences.
Third paragraph (optional): Clinical interpretation - what this might mean in practice.
Fourth paragraph (optional): Important caveats or limitations.

Keep it concise and structured. Do not use markdown headers, numbered lists, or bullet points."""
            
            elif result_type == "ate":
                X = result_data.get("X", "treatment")
                Y = result_data.get("Y", "outcome")
                ate = result_data.get("ate", 0)
                y_treated_mean = result_data.get("y_treated_mean", 0)
                y_control_mean = result_data.get("y_control_mean", 0)
                y_treated_std = result_data.get("y_treated_std", 0)
                y_control_std = result_data.get("y_control_std", 0)
                x_treated = result_data.get("x_treated", 1)
                x_control = result_data.get("x_control", 0)
                
                usr_msg = f"""A physician asked: "What is the effect of {X} on {Y}?"

The analysis computed the Average Treatment Effect (ATE) - this is like comparing two groups in a randomized trial.

Results:
- ATE = {ate:.4f}
- When {X} = {x_treated} (treatment group): {Y} has mean = {y_treated_mean:.4f} (std: {y_treated_std:.4f})
- When {X} = {x_control} (control group): {Y} has mean = {y_control_mean:.4f} (std: {y_control_std:.4f})

{var_context}
{context}

Explain what these results mean in simple terms for a medical doctor. Write 2-4 short, focused paragraphs (maximum 150 words total) separated by blank lines. Each paragraph should cover one key point.

First paragraph: Explain what ATE means and the key finding (the effect size and direction - positive or negative).
Second paragraph: Explain the magnitude of the effect and what the standard deviations tell us about variability.
Third paragraph (optional): Clinical interpretation - what this means in practice and practical significance.
Fourth paragraph (optional): Important limitations or caveats.

Keep it concise and structured. Do not use markdown headers, numbered lists, or bullet points."""
            
            elif result_type == "association":
                corr_matrix = result_data.get("correlation_matrix", {})
                
                usr_msg = f"""A physician asked to see associations between variables.

The analysis computed correlations (how variables move together).

Correlation Matrix:
{corr_matrix}

{var_context}
{context}

Explain what these correlations mean in simple terms for a medical doctor. Write 2-4 short, focused paragraphs (maximum 150 words total) separated by blank lines. Each paragraph should cover one key point.

First paragraph: Explain what correlation means and identify the strongest associations (which variables are most related).
Second paragraph: Explain the difference between positive and negative correlations and what weak associations mean.
Third paragraph (optional): Important warning that correlation does NOT mean causation.
Fourth paragraph (optional): What this means for clinical understanding.

Keep it concise and structured. Do not use markdown headers, numbered lists, or bullet points."""
            
            else:
                # Generic explanation
                usr_msg = f"""A physician received these analysis results:

{result_data}

{var_context}
{context}

Explain what these results mean in simple terms for a medical doctor."""
            
            try:
                llm_response = llm_client.chat.completions.create(
                    model=llm_model_interpretation,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": usr_msg}
                    ],
                    temperature=0.3,
                    timeout=30
                )
                
                explanation = llm_response.choices[0].message.content.strip()
                
                # Clean up markdown formatting for clean, scientific output
                explanation = self._clean_explanation_formatting(explanation)
                
                return explanation
            except Exception as e:
                print(f"[WARNING] Failed to generate LLM explanation: {e}")
                return ""  # Return empty if LLM fails - don't break the flow
        except Exception as e:
            print(f"[WARNING] Error in _generate_medical_explanation: {e}")
            return ""  # Return empty if anything fails
    
    def _clean_explanation_formatting(self, text: str) -> str:
        """
        Clean up markdown formatting from LLM explanations to make them clean and scientific.
        Removes headers, excessive spacing, and converts to clean paragraph format.
        """
        if not text:
            return text
        
        # Remove markdown headers (###, ##, #) - convert to plain text or remove
        text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
        
        # Remove section headers that look like "4. Clinical Interpretation" or "### 4. Clinical Interpretation"
        text = re.sub(r'^(?:#{1,6}\s+)?\d+\.\s+[A-Z][^:]+:\s*$', '', text, flags=re.MULTILINE)
        
        # Remove numbered list markers at start of lines (like "1.", "2.", "3.", etc.)
        # Convert them to regular sentences
        text = re.sub(r'^(\d+)\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove bullet points (-, *, •)
        text = re.sub(r'^[\-\*•]\s+', '', text, flags=re.MULTILINE)
        
        # Preserve paragraph breaks (2 newlines) but remove excessive blank lines (more than 2 consecutive)
        # First, normalize to max 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove bold/italic markdown but keep the text
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** -> bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # *italic* -> italic
        
        # Remove markdown links but keep the text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Clean up spacing around colons and periods
        text = re.sub(r'\s+:\s+', ': ', text)
        text = re.sub(r'\s+\.\s+', '. ', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        # Preserve paragraph breaks (empty lines between paragraphs) but remove excessive empty lines
        # Keep single empty lines (paragraph breaks) but remove multiple consecutive empty lines
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                # Keep one empty line as paragraph break
                cleaned_lines.append('')
                prev_empty = True
            # Skip additional empty lines (prev_empty is already True)
        text = '\n'.join(cleaned_lines)
        
        # Remove empty lines at start and end
        text = text.strip()
        
        # Ensure proper sentence spacing (normalize to single space after periods)
        text = re.sub(r'\.\s+', '. ', text)
        
        # Remove any remaining markdown artifacts
        text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove code backticks
        
        # Convert multiple newlines to paragraph breaks (max 2 newlines)
        text = re.sub(r'\n\n+', '\n\n', text)
        
        return text

    def _default_ate_interpretation(self, result_data: Dict[str, Any]) -> str:
        """Fallback ATE interpretation when LLM output is unavailable."""
        X = result_data.get("X", "the treatment")
        Y = result_data.get("Y", "the outcome")
        ate = float(result_data.get("ate", 0.0) or 0.0)
        y_treated_mean = float(result_data.get("y_treated_mean", 0.0) or 0.0)
        y_control_mean = float(result_data.get("y_control_mean", 0.0) or 0.0)

        direction = "increase" if ate >= 0 else "decrease"
        magnitude = abs(ate)

        return (
            f"When changing {X}, the expected value of {Y} changes by about {magnitude:.4f} on average "
            f"({direction} direction in this model).\n\n"
            f"In this run, the treated setting has mean {y_treated_mean:.4f} and the control setting has "
            f"mean {y_control_mean:.4f} for {Y}. This is a model-based causal estimate and should be interpreted "
            f"together with DAG assumptions and CI test quality."
        )

    def _calculation_followup_note(self) -> str:
        """Post-calculation guidance for deeper follow-up questions."""
        return (
            "\nYou can now ask follow-up questions about these calculations (for example: "
            "`explain this effect in detail`, `compare treated vs control`, or "
            "`what is clinically important here?`). "
            f"I will use the interpretation model (`{llm_model_interpretation}`), which reviews the returned values "
            "and available calculation context from your session for a deeper explanation.\n"
        )

    def _build_interpretation_context(self) -> Dict[str, Any]:
        """Compact state snapshot used for no-tool interpretation answers."""
        session = sessions.get(self.session_id, {})
        query_results = session.get("query_results", {}) or {}
        ate_result = query_results.get("ate")
        ci_results = self.state.ci_test_results or session.get("ci_test_results")

        context: Dict[str, Any] = {
            "current_step": self.state.current_step,
            "has_data": self.state.data_path is not None,
            "has_dag": self.state.proposed_dag is not None,
            "has_fitted_model": self.state.fitted_model is not None,
            "experiment_dir": self.state.experiment_dir or session.get("experiment_dir"),
            "dag_variables": self.state.proposed_dag.get("variables", []) if self.state.proposed_dag else [],
            "dag_edges_count": len(self.state.proposed_dag.get("edges", [])) if self.state.proposed_dag else 0,
            "has_ci_test_results": bool(ci_results),
            "has_ate_result": bool(ate_result),
        }

        if ci_results:
            context["ci_consistent"] = ci_results.get("consistent")
            context["ci_rejected_count"] = ci_results.get("rejected_count")

        if ate_result:
            context["latest_ate"] = {
                "X": ate_result.get("X"),
                "Y": ate_result.get("Y"),
                "ate": ate_result.get("ate"),
                "x_treated": ate_result.get("x_treated"),
                "x_control": ate_result.get("x_control"),
                "y_treated_mean": ate_result.get("y_treated_mean"),
                "y_control_mean": ate_result.get("y_control_mean"),
            }

        return context

    async def _answer_general_question_with_interpretation_model(
        self,
        user_message: str,
        decision_model_draft: Optional[str] = None,
    ) -> str:
        """
        Answer no-tool questions with the interpretation model and analysis context.
        Falls back to decision-model draft if interpretation call fails.
        """
        context = self._build_interpretation_context()
        sys_msg = (
            "You are the deep-interpretation assistant for a causal analysis chatbot. "
            "The user may ask conceptual questions or questions about already-computed results. "
            "Use the provided context/results, be accurate and concise, and do not invent numbers. "
            "If a requested value is not available, say what is missing and suggest the next concrete command."
        )
        usr_msg = (
            f"SESSION CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
            f"USER QUESTION:\n{user_message}\n\n"
            "Answer in clear prose for a non-technical user."
        )
        try:
            llm_response = llm_client.chat.completions.create(
                model=llm_model_interpretation,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg},
                ],
                temperature=0.2,
                timeout=30,
            )
            text = (llm_response.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:
            print(f"[WARNING] Interpretation-model no-tool answer failed: {e}")

        if decision_model_draft and str(decision_model_draft).strip():
            return str(decision_model_draft).strip()
        return self._get_fallback_response_for_query(user_message)
    
    def _get_fallback_response_for_query(self, query: str) -> str:
        """Provide a helpful fallback response when query can't be processed"""
        response = "**I'm not sure how to handle that request.**\n\n"
        response += f"**Your query:** {query}\n\n"
        response += "**Here's what I can help you with:**\n\n"
        
        dag_vars = self.state.proposed_dag.get("variables", []) if self.state.proposed_dag else []
        if not dag_vars:
            dag_vars = self._example_vars()
        example_var = dag_vars[0] if dag_vars else "x1"
        example_outcome = dag_vars[-1] if len(dag_vars) > 1 else example_var
        
        actions = [
            {
                "label": "Compute Average Treatment Effect (ATE)",
                "command": "compute the ate",
                "description": "I will ask which treatment (X) and outcome (Y) variables to use"
            },
            {
                "label": "Generate plots and visualizations",
                "command": "sample 1000 data points and show me the plots",
                "description": "Create distribution plots, loss history, and DAG visualization"
            },
            {
                "label": "Download full report",
                "command": "generate report",
                "description": "Get a comprehensive PDF report with all results and plots"
            }
        ]
        response += self._format_action_suggestions(actions)
        return response
    
    def _format_action_suggestions(self, actions: List[Dict[str, str]]) -> str:
        """
        Format action suggestions for display in chat UI
        
        Args:
            actions: List of dicts with 'label' (what user sees) and 'command' (what to send)
        
        Returns:
            Formatted string with action suggestions
        """
        if not actions:
            return ""
        
        response = "\n\n---\n\n**What would you like to do next?**\n\n"
        for i, action in enumerate(actions, 1):
            response += f"{i}. **{action['label']}**\n"
            response += f"   → Type: `{action['command']}`\n"
            if 'description' in action:
                response += f"   {action['description']}\n"
            response += "\n"
        
        response += "**Tip:** You can also type your request in natural language, and I'll understand it!"
        return response

    def _store_mcp_plot_urls(self, artifact_manifest: List[Dict[str, Any]], imported_urls: List[str]) -> None:
        if not artifact_manifest or not imported_urls:
            return
        if self.session_id not in sessions:
            sessions[self.session_id] = {"session_id": self.session_id, "messages": []}
        store = sessions[self.session_id].setdefault("mcp_plot_urls_by_kind", {})
        for idx, entry in enumerate(artifact_manifest):
            kind = str(entry.get("kind", "")).strip()
            if not kind:
                continue
            if idx < len(imported_urls):
                store[kind] = imported_urls[idx]

    def _get_stored_mcp_plot_urls(self, kinds: List[str]) -> List[str]:
        store = sessions.get(self.session_id, {}).get("mcp_plot_urls_by_kind", {})
        urls: List[str] = []
        for kind in kinds:
            url = store.get(kind)
            if url:
                urls.append(url)
        return urls

    def _import_and_store_mcp_artifacts(self, artifact_paths: List[str], artifact_manifest: List[Dict[str, Any]]) -> List[str]:
        urls = CausalTools.import_mcp_artifacts(self.session_id, artifact_paths or [])
        if urls:
            self._store_mcp_plot_urls(artifact_manifest or [], urls)
        return urls

    def _ensure_session_record(self) -> Dict[str, Any]:
        if self.session_id not in sessions:
            sessions[self.session_id] = {"session_id": self.session_id, "messages": []}
        return sessions[self.session_id]

    def _set_session_ate_result(self, ate_result: Dict[str, Any]) -> None:
        session = self._ensure_session_record()
        session.setdefault("query_results", {})
        session["query_results"]["ate"] = ate_result

    def _set_session_report_path(self, report_path: Path) -> None:
        session = self._ensure_session_record()
        session["report_path"] = str(report_path)

    def _invalidate_runtime_after_dag_change(self) -> None:
        self.state.fitted_model = None
        self.state.experiment_dir = None
        self.state.ci_test_results = None
        self.state.pending_tool = None
        self.state.pending_tool_args = None
        self.state.pending_missing_param = None
        session = self._ensure_session_record()
        invalidate_session_after_dag_update(session)

    def _parameter_defaults(self) -> Dict[str, Any]:
        return {
            "alpha": 0.05,
            "epochs": 100,
            "learning_rate": 0.01,
            "batch_size": 512,
            "n_samples": 10000,
            "x_treated": 1.0,
            "x_control": 0.0,
            "expert_text": "",
        }

    def _required_or_guided_param_order(self, tool_name: str) -> List[str]:
        # Only include parameters that truly require guided user input.
        # Most tools have safe defaults and should execute directly.
        if tool_name == "propose_dag":
            return ["vars"]
        if tool_name == "fit_model":
            return ["epochs", "learning_rate", "batch_size"]
        if tool_name == "compute_ate":
            return ["X", "Y"]
        return []

    def _first_missing_param(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[str]:
        ordered = self._required_or_guided_param_order(tool_name)
        for param in ordered:
            value = tool_args.get(param)
            if value is None:
                return param
            if param == "expert_text" and value == "":
                continue
            if isinstance(value, str) and not value.strip():
                return param
            if isinstance(value, list) and len(value) == 0:
                return param
        return None

    def _session_variables(self) -> List[str]:
        if self.state.proposed_dag and self.state.proposed_dag.get("variables"):
            return list(self.state.proposed_dag.get("variables", []))
        if self.state.data_df is not None:
            return list(self.state.data_df.columns)
        return []

    def _example_vars(self) -> List[str]:
        vars_list = self._session_variables()
        if vars_list:
            return vars_list
        return ["treatment_variable", "outcome_variable"]

    def _quick_pick_actions_for_param(self, tool_name: str, missing_param: str, partial_args: Dict[str, Any]) -> List[Dict[str, str]]:
        defaults = self._parameter_defaults()
        actions: List[Dict[str, str]] = []
        variables = self._session_variables()

        if missing_param in ("X", "Y"):
            role = "treatment" if missing_param == "X" else "outcome"
            for v in variables[:6]:
                actions.append({
                    "label": f"Set {role} = {v}",
                    "command": v,
                    "description": f"Use `{v}` as the {role} variable"
                })
            return actions

        if missing_param == "alpha":
            for val in [0.01, 0.05, 0.10]:
                actions.append({
                    "label": f"Use alpha = {val}",
                    "command": str(val),
                    "description": "Significance level for CI testing"
                })
        elif missing_param == "epochs":
            for val in [50, 100, 200]:
                actions.append({
                    "label": f"Use {val} epochs",
                    "command": str(val),
                    "description": "Training duration for model fitting"
                })
            actions.append({
                "label": "Use all default fit settings",
                "command": "use defaults",
                "description": "Applies epochs=100, learning_rate=0.01, batch_size=512"
            })
        elif missing_param == "learning_rate":
            for val in [0.001, 0.01, 0.05]:
                actions.append({
                    "label": f"Use learning rate = {val}",
                    "command": str(val),
                    "description": "Optimizer step size"
                })
        elif missing_param == "batch_size":
            for val in [256, 512, 1024]:
                actions.append({
                    "label": f"Use batch size = {val}",
                    "command": str(val),
                    "description": "Mini-batch size during fitting"
                })
        elif missing_param == "n_samples":
            for val in [1000, 5000, 10000]:
                actions.append({
                    "label": f"Sample {val} points",
                    "command": str(val),
                    "description": "Number of synthetic draws"
                })
        elif missing_param in ("x_treated", "x_control"):
            role = "treated" if missing_param == "x_treated" else "control"
            for val in [0.0, 1.0, 2.0]:
                actions.append({
                    "label": f"Set {role} value = {val}",
                    "command": str(val),
                    "description": "Intervention value for ATE computation"
                })
        elif missing_param == "intervention_choice":
            actions.extend([
                {
                    "label": "Open treatment plot explorer",
                    "command": "yes, I need support",
                    "description": "Open the TRAM-style treatment distribution plot to pick values"
                },
                {
                    "label": "No support, I will set values directly",
                    "command": "treated 1.0, control 0.0",
                    "description": "Provide treated/control values directly in chat"
                },
                {
                    "label": "Use default intervention values",
                    "command": "use defaults",
                    "description": "Use x_treated=1.0 and x_control=0.0"
                },
            ])
        elif missing_param == "intervention_confirm":
            actions.extend([
                {
                    "label": "Run ATE with these values",
                    "command": "run ate",
                    "description": "Execute ATE with the selected treated/control values"
                },
                {
                    "label": "Change values in interactive chooser",
                    "command": "sample 1000 data points and show me the plots",
                    "description": "Re-open intervention chooser and adjust values"
                },
                {
                    "label": "Cancel",
                    "command": "cancel",
                    "description": "Abort this ATE request"
                },
            ])
        elif missing_param == "expert_text":
            actions.append({
                "label": "Skip expert constraints",
                "command": "skip",
                "description": "Continue DAG proposal without domain notes"
            })
            actions.append({
                "label": "Add expert knowledge",
                "command": "Smoking causes LungCancer",
                "description": "Example causal prior text"
            })

        if missing_param in defaults and missing_param not in ("X", "Y", "expert_text"):
            actions.append({
                "label": f"Use default ({defaults[missing_param]})",
                "command": "default",
                "description": "Accept the standard value"
            })
        actions.append({
            "label": "Cancel this pending action",
            "command": "cancel",
            "description": "Clear this prompt and switch tasks"
        })
        return actions

    def _guided_prompt_for_missing_param(self, tool_name: str, missing_param: str, partial_args: Dict[str, Any]) -> Dict[str, Any]:
        formats = {
            "X": "Must be one of your dataset variables.",
            "Y": "Must be one of your dataset variables.",
            "alpha": "Float between 0 and 1 (e.g., 0.05).",
            "epochs": "Positive integer (e.g., 100).",
            "learning_rate": "Positive float (e.g., 0.01).",
            "batch_size": "Positive integer (e.g., 512).",
            "n_samples": "Positive integer (e.g., 10000).",
            "x_treated": "Numeric intervention value (e.g., 1.0).",
            "x_control": "Numeric intervention value (e.g., 0.0).",
            "expert_text": "Optional domain knowledge text. Type `skip` to continue without it.",
            "vars": "List of variable names from your uploaded dataset.",
            "intervention_choice": "Choose whether you need support; then provide treated/control values or open interactive plots.",
            "intervention_confirm": "Confirm whether to execute ATE with the selected values.",
        }
        examples = {
            "X": "BMI",
            "Y": "BloodPressure",
            "alpha": "0.05",
            "epochs": "100",
            "learning_rate": "0.01",
            "batch_size": "512",
            "n_samples": "10000",
            "x_treated": "1.0",
            "x_control": "0.0",
            "expert_text": "Smoking causes LungCancer",
            "vars": "age, bmi, systolic_bp",
            "intervention_choice": "yes, I need support",
            "intervention_confirm": "run ate",
        }

        if tool_name == "compute_ate" and missing_param in {"X", "Y"}:
            if missing_param == "X" and partial_args.get("X") and partial_args.get("Y"):
                prompt = (
                    "**Guided ATE setup**\n\n"
                    f"I detected a possible pair: treatment **{partial_args.get('X')}**, outcome **{partial_args.get('Y')}**.\n\n"
                    "Before I compute ATE, please confirm or change the treatment variable (X).\n"
                    f"- Example input: `{examples.get('X', 'X')}`\n"
                    "Then I will ask for the outcome variable (Y)."
                )
            else:
                prompt = (
                    "**Guided ATE setup**\n\n"
                    "Please choose your variables first:\n"
                    f"- Treatment variable (X): **{partial_args.get('X') or '<missing>'}**\n"
                    f"- Outcome variable (Y): **{partial_args.get('Y') or '<missing>'}**\n\n"
                    f"- Missing parameter right now: **{missing_param}**\n"
                    f"- Expected format: {formats.get(missing_param, 'Provide a valid value.')}\n"
                    f"- Example input: `{examples.get(missing_param, 'value')}`\n\n"
                    "Tip: you can provide both at once in one message, e.g. `X bmi, Y blood_pressure`.\n"
                    "Then I will suggest opening the treatment plot explorer to pick intervention values."
                )
        elif tool_name == "compute_ate" and missing_param == "intervention_choice":
            prompt = (
                "**Guided ATE setup — intervention values**\n\n"
                f"- Treatment variable (X): **{partial_args.get('X', '<missing>')}**\n"
                f"- Outcome variable (Y): **{partial_args.get('Y', '<missing>')}**\n\n"
                "Do you need support choosing intervention values?\n"
                "- Recommended: reply `yes, I need support` to open the treatment plot explorer.\n"
                "- Or reply directly with `treated <value>, control <value>`.\n"
                "- Or type `use defaults` to use treated=1.0 and control=0.0."
            )
        else:
            prompt = (
                f"**Guided input needed for `{tool_name}`**\n\n"
                f"- Missing parameter: **{missing_param}**\n"
                f"- Expected format: {formats.get(missing_param, 'Provide a valid value.')}\n"
            )
            if tool_name != "propose_dag":
                prompt += f"- Example input: `{examples.get(missing_param, 'value')}`\n"
        if partial_args:
            prompt += f"\n- Current parameters: `{json.dumps(partial_args)}`\n"

        return {
            "_guided_prompt": prompt,
            "_guided_actions": self._quick_pick_actions_for_param(tool_name, missing_param, partial_args),
        }

    def _build_ate_preview_prompt(self, args: Dict[str, Any]) -> str:
        x_var = str(args.get("X", "X"))
        y_var = str(args.get("Y", "Y"))
        treated = float(args.get("x_treated", 1.0))
        control = float(args.get("x_control", 0.0))
        delta = treated - control
        preview = (
            "**ATE Preview Card**\n\n"
            f"- Treatment variable (X): **{x_var}**\n"
            f"- Outcome variable (Y): **{y_var}**\n"
            f"- Selected treated value: **{treated:.6g}**\n"
            f"- Selected control value: **{control:.6g}**\n"
            f"- Intervention delta (treated - control): **{delta:.6g}**\n"
        )

        if self.state.data_df is not None and x_var in self.state.data_df.columns:
            vals = pd.to_numeric(self.state.data_df[x_var], errors="coerce").dropna()
            if len(vals) > 0:
                p10 = float(np.percentile(vals, 10))
                p50 = float(np.percentile(vals, 50))
                p90 = float(np.percentile(vals, 90))
                scale = abs(p90 - p10) if abs(p90 - p10) > 1e-12 else 1.0
                contrast_ratio = abs(delta) / scale
                if contrast_ratio < 0.25:
                    contrast_label = "small"
                elif contrast_ratio < 0.75:
                    contrast_label = "moderate"
                else:
                    contrast_label = "large"
                preview += (
                    f"- Data percentiles for `{x_var}`: p10={p10:.4g}, p50={p50:.4g}, p90={p90:.4g}\n"
                    f"- Expected contrast size vs percentile span: **{contrast_label}**\n"
                )
        preview += "\nConfirm to run ATE, or adjust values first."
        return preview

    def _set_pending_tool(self, tool_name: str, tool_args: Dict[str, Any], missing_param: str) -> None:
        self.state.pending_tool = tool_name
        self.state.pending_tool_args = dict(tool_args)
        self.state.pending_missing_param = missing_param
        self.save_session_state()

    def _clear_pending_tool(self) -> None:
        self.state.pending_tool = None
        self.state.pending_tool_args = None
        self.state.pending_missing_param = None
        self.save_session_state()

    def _pending_tool_context_valid(self, pending_tool: str) -> bool:
        """
        Guard against stale pending prompts after workflow step changes.
        """
        step = self.state.current_step or "initial"
        if pending_tool == "fit_model":
            # fit_model can be legitimately pending while still at dag_proposed
            # (guided hyperparameter intake before actual fit execution).
            return (
                step in {"dag_proposed", "dag_tested", "dag_finalized", "model_fitted"}
                and self.state.proposed_dag is not None
                and self.state.data_path is not None
            )
        if pending_tool == "test_dag":
            return self.state.proposed_dag is not None and self.state.data_path is not None
        if pending_tool in {"sample", "compute_ate", "create_plots"}:
            return self.state.experiment_dir is not None
        if pending_tool == "propose_dag":
            return self.state.data_df is not None
        return True

    def _parse_pending_param_value(self, param: str, user_message: str) -> Optional[Any]:
        text = (user_message or "").strip()
        if not text:
            return None

        if text.lower() in {"skip", "none", "no"} and param == "expert_text":
            return ""

        if param in {"X", "Y"}:
            variables = self._session_variables()
            if not variables:
                return None
            for v in variables:
                if text.lower() == str(v).lower():
                    return v
            for v in variables:
                if re.search(rf"\b{re.escape(str(v))}\b", text, flags=re.IGNORECASE):
                    return v
            return None

        if param == "vars":
            parts = [p.strip() for p in re.split(r"[,\s]+", text) if p.strip()]
            return parts if parts else None

        if param == "expert_text":
            return text

        number_match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not number_match:
            return None
        value_str = number_match.group(0)

        int_params = {"epochs", "batch_size", "n_samples"}
        if param in int_params:
            value = int(float(value_str))
            return value if value > 0 else None

        value = float(value_str)
        if param == "alpha":
            return value if 0 < value <= 1 else None
        return value

    def _maybe_prompt_missing_params(self, tool_name: str, tool_args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if tool_name == "propose_dag":
            vars_list = tool_args.get("vars", [])
            if not vars_list and self.state.data_df is not None:
                tool_args["vars"] = list(self.state.data_df.columns)
                vars_list = tool_args["vars"]
            if not vars_list:
                payload = self._guided_prompt_for_missing_param(tool_name, "vars", tool_args)
                self._set_pending_tool(tool_name, tool_args, "vars")
                return {"tool": tool_name, "success": True, "data": payload, "error": None, "plots": []}

        if tool_name == "compute_ate":
            X = tool_args.get("X")
            Y = tool_args.get("Y")
            has_treated = tool_args.get("x_treated") is not None
            has_control = tool_args.get("x_control") is not None
            vars_confirmed = bool(tool_args.get("_vars_confirmed"))

            # Always ask user to select/confirm ATE variables first (X then Y),
            # even if upstream parsing suggested a pair.
            if not vars_confirmed:
                if not X:
                    payload = self._guided_prompt_for_missing_param(tool_name, "X", tool_args)
                    self._set_pending_tool(tool_name, tool_args, "X")
                    return {"tool": tool_name, "success": True, "data": payload, "error": None, "plots": []}
                if not Y:
                    payload = self._guided_prompt_for_missing_param(tool_name, "Y", tool_args)
                    self._set_pending_tool(tool_name, tool_args, "Y")
                    return {"tool": tool_name, "success": True, "data": payload, "error": None, "plots": []}
                # If both were present but not explicitly user-confirmed in this run,
                # ask to confirm by re-selecting X first.
                payload = self._guided_prompt_for_missing_param(tool_name, "X", tool_args)
                self._set_pending_tool(tool_name, tool_args, "X")
                return {"tool": tool_name, "success": True, "data": payload, "error": None, "plots": []}

            if X and Y and not has_treated and not has_control and not tool_args.get("_intervention_choice_done"):
                payload = self._guided_prompt_for_missing_param(tool_name, "intervention_choice", tool_args)
                self._set_pending_tool(tool_name, tool_args, "intervention_choice")
                return {"tool": tool_name, "success": True, "data": payload, "error": None, "plots": []}

        missing_param = self._first_missing_param(tool_name, tool_args)
        if missing_param is None:
            return None

        payload = self._guided_prompt_for_missing_param(tool_name, missing_param, tool_args)
        self._set_pending_tool(tool_name, tool_args, missing_param)
        return {"tool": tool_name, "success": True, "data": payload, "error": None, "plots": []}

    async def _try_handle_pending_tool(self, user_message: str) -> Optional[str]:
        if not self.state.pending_tool or not self.state.pending_missing_param:
            return None

        msg = (user_message or "").strip()
        normalized_msg = re.sub(r"[^\w\s]", "", msg.lower()).strip()

        if normalized_msg in {"cancel", "cancel pending", "stop"}:
            self._clear_pending_tool()
            return "Pending guided action was cancelled. You can continue with any other request."

        pending_tool = self.state.pending_tool
        pending_args = dict(self.state.pending_tool_args or {})
        missing_param = self.state.pending_missing_param
        handled_pending_choice = False

        if pending_tool == "compute_ate" and missing_param == "intervention_confirm":
            if normalized_msg in {"run ate", "confirm", "yes", "calculate ate", "execute"}:
                self._clear_pending_tool()
                result = await self._execute_tool(pending_tool, pending_args)
                return await self._format_tool_results([(pending_tool, result)], user_message)
            if any(k in normalized_msg for k in ["change", "adjust", "plot", "sample", "chooser"]):
                self._set_pending_tool("compute_ate", pending_args, "intervention_choice")
                response = "<!-- OPEN_INTERVENTION_CHOOSER -->\n"
                response += "**Intervention chooser reopened.**\n\nSelect new values and click **Apply values**."
                return response
            return None

        if pending_tool == "compute_ate" and missing_param == "intervention_choice":
            if any(k in normalized_msg for k in ["sample", "plot", "distribution", "support", "help"]) or normalized_msg in {"yes", "yes i need support", "i need support"}:
                x_var = pending_args.get("X", "")
                response = "<!-- OPEN_INTERVENTION_CHOOSER -->\n"
                response += "**Intervention value chooser**\n\n"
                if x_var:
                    response += f"I opened an interactive plot for `{x_var}` so you can choose intervention values visually.\n\n"
                response += "After selecting values, click **Apply values** to run ATE with your chosen treated/control settings.\n\n"
                actions = [
                    {
                        "label": "Use defaults instead",
                        "command": "use defaults",
                        "description": "Set treated=1.0 and control=0.0 immediately"
                    },
                    {
                        "label": "Need help choosing values",
                        "command": "help me choose intervention values",
                        "description": "Get percentile-based suggestions from the data"
                    },
                    {
                        "label": "Cancel ATE setup",
                        "command": "cancel",
                        "description": "Exit intervention-value selection"
                    },
                ]
                response += self._format_action_suggestions(actions)
                return response
            if any(k in normalized_msg for k in ["no support", "without support", "dont need support", "do not need support", "no help"]):
                prompt = (
                    "**Direct intervention input selected**\n\n"
                    "Please provide your intervention values as:\n"
                    "`treated <value>, control <value>`\n\n"
                    "Example: `treated 2.5, control 1.0`"
                )
                return prompt
            if any(k in normalized_msg for k in ["help", "choose", "suggest"]):
                x_var = pending_args.get("X")
                advice = "**Intervention value support**\n\n"
                advice += "To choose intervention values, compare low vs high plausible values for your treatment.\n"
                if x_var and self.state.data_df is not None and x_var in self.state.data_df.columns:
                    vals = pd.to_numeric(self.state.data_df[x_var], errors="coerce").dropna()
                    if len(vals) > 0:
                        p25 = float(np.percentile(vals, 25))
                        p50 = float(np.percentile(vals, 50))
                        p75 = float(np.percentile(vals, 75))
                        actions = [
                            {
                                "label": f"Use low vs high ({p25:.3f} -> {p75:.3f})",
                                "command": f"treated {p75:.3f}, control {p25:.3f}",
                                "description": "Contrast upper and lower quartiles"
                            },
                            {
                                "label": f"Use median vs low ({p50:.3f} -> {p25:.3f})",
                                "command": f"treated {p50:.3f}, control {p25:.3f}",
                                "description": "Moderate intervention contrast"
                            },
                            {
                                "label": "Use defaults",
                                "command": "use defaults",
                                "description": "Set treated=1.0 and control=0.0"
                            },
                            {
                                "label": "Show plots first",
                                "command": "sample 1000 data points and show me the plots",
                                "description": "Inspect variable distributions before choosing"
                            },
                        ]
                        advice += f"\nDetected `{x_var}` distribution percentiles: p25={p25:.3f}, p50={p50:.3f}, p75={p75:.3f}\n"
                        advice += self._format_action_suggestions(actions)
                        return advice
                advice += "\nReply with values like: `treated 1.0, control 0.0`, or type `use defaults`."
                return advice

            if normalized_msg in {"default", "use default", "use defaults"}:
                pending_args["x_treated"] = self._parameter_defaults()["x_treated"]
                pending_args["x_control"] = self._parameter_defaults()["x_control"]
                pending_args["_intervention_choice_done"] = True
                handled_pending_choice = True
            else:
                pair_match = re.search(r"treated\s*=?\s*(-?\d+(?:\.\d+)?)\s*[,;]?\s*control\s*=?\s*(-?\d+(?:\.\d+)?)", msg.lower())
                if pair_match:
                    pending_args["x_treated"] = float(pair_match.group(1))
                    pending_args["x_control"] = float(pair_match.group(2))
                    pending_args["_intervention_choice_done"] = True
                    handled_pending_choice = True
                else:
                    return None

            if handled_pending_choice:
                pending_args["_intervention_choice_done"] = True
                self._set_pending_tool("compute_ate", pending_args, "intervention_confirm")
                preview_data = {
                    "_guided_prompt": self._build_ate_preview_prompt(pending_args),
                    "_guided_actions": self._quick_pick_actions_for_param("compute_ate", "intervention_confirm", pending_args),
                }
                return await self._format_tool_results([(pending_tool, {"success": True, "data": preview_data, "plots": []})], user_message)

        # Clear stale pending prompts before applying shorthand defaults.
        if not handled_pending_choice and not self._pending_tool_context_valid(pending_tool):
            self._clear_pending_tool()
            return None

        # Keep "default/use defaults" bound to the active pending prompt even if
        # current_step drifts temporarily due session sync timing.
        if normalized_msg in {"default", "use default", "use defaults"} and not handled_pending_choice:
            defaults = self._parameter_defaults()
            if missing_param in defaults:
                pending_args[missing_param] = defaults[missing_param]
                if pending_tool == "fit_model" and normalized_msg in {"use default", "use defaults"}:
                    pending_args.setdefault("epochs", defaults["epochs"])
                    pending_args.setdefault("learning_rate", defaults["learning_rate"])
                    pending_args.setdefault("batch_size", defaults["batch_size"])
            else:
                return None
        else:
            if pending_tool == "compute_ate" and missing_param in {"X", "Y"}:
                variables = self._session_variables()
                mentioned: List[str] = []
                lowered = msg.lower()
                for var in variables:
                    if re.search(rf"\b{re.escape(str(var).lower())}\b", lowered):
                        mentioned.append(var)
                # Allow one-shot input like "X=bmi, Y=bp" while pending for X.
                if missing_param == "X" and len(mentioned) >= 2:
                    pending_args["X"] = mentioned[0]
                    pending_args["Y"] = next((v for v in mentioned[1:] if v != mentioned[0]), mentioned[1])
                    pending_args["_vars_confirmed"] = True
                elif missing_param == "Y" and len(mentioned) >= 1:
                    pending_args["Y"] = next((v for v in mentioned if v != pending_args.get("X")), mentioned[0])
                    pending_args["_vars_confirmed"] = True
                elif len(mentioned) == 1:
                    pending_args[missing_param] = mentioned[0]
                    if missing_param == "Y":
                        pending_args["_vars_confirmed"] = True
                else:
                    value = self._parse_pending_param_value(missing_param, msg)
                    if value is None:
                        return None
                    pending_args[missing_param] = value
                    if missing_param == "Y":
                        pending_args["_vars_confirmed"] = True
            else:
                value = self._parse_pending_param_value(missing_param, msg)
                if value is None:
                    return None
                pending_args[missing_param] = value

        next_missing = self._first_missing_param(pending_tool, pending_args)
        if next_missing is not None:
            self._set_pending_tool(pending_tool, pending_args, next_missing)
            data = self._guided_prompt_for_missing_param(pending_tool, next_missing, pending_args)
            return await self._format_tool_results([(pending_tool, {"success": True, "data": data, "plots": []})], user_message)

        self._clear_pending_tool()
        result = await self._execute_tool(pending_tool, pending_args)
        return await self._format_tool_results([(pending_tool, result)], user_message)
    
    def save_session_state(self):
        """Save current orchestrator state to session storage - call after any state change"""
        if self.session_id not in sessions:
            sessions[self.session_id] = {
                "session_id": self.session_id,
                "messages": [],
                "created_at": datetime.now().isoformat()
            }
        
        # Update session with current state
        sessions[self.session_id].update({
            "data_path": self.state.data_path,
            "data_df": self.state.data_df.to_dict('records') if self.state.data_df is not None else None,
            "proposed_dag": self.state.proposed_dag,
            "ci_test_results": self.state.ci_test_results,
            "fitted_model": self.state.fitted_model,
            "experiment_dir": self.state.experiment_dir,
            "current_step": self.state.current_step,  # CRITICAL: Always save current_step
            "pending_tool": self.state.pending_tool,
            "pending_tool_args": self.state.pending_tool_args,
            "pending_missing_param": self.state.pending_missing_param,
            "messages": [msg.model_dump() if hasattr(msg, 'model_dump') else (msg.dict() if hasattr(msg, 'dict') else msg) for msg in self.state.messages]
        })
        print(f"[DEBUG] Saved state for session {self.session_id}: current_step='{self.state.current_step}'")
    
    async def process_message(self, user_message: str) -> str:
        """
        Main entry point: process a user message via the AI Agent.
        
        Architecture:
            Chat UI  →  user message (natural language)
                     →  AI Agent (LLM with function calling)
                     →  LLM selects tool from ToolRegistry
                     →  Tool executes locally (CausalTools)
                     →  Agent formats result + updates state
                     →  Response sent back to Chat UI
        """
        # Add user message to history
        self.state.messages.append(ChatMessage(
            role="user",
            content=user_message,
            timestamp=datetime.now().isoformat()
        ))
        
        print(f"[AGENT] Processing message: '{user_message[:80]}...'")
        print(f"[AGENT] Current step: '{self.state.current_step}'")

        # Continue guided parameter intake if a prior tool call is pending.
        pending_response = await self._try_handle_pending_tool(user_message)
        if pending_response is not None:
            return pending_response

        # ── Pre-route: intercept deterministic workflow commands ──
        # These are commands the chatbot itself suggested via action buttons.
        # They must be handled by the workflow engine, NOT the LLM, because
        # there is no matching tool in ToolRegistry (they are multi-step
        # orchestration actions, not single statistical tools).
        pre_route = self._try_deterministic_route(user_message)
        if pre_route is not None:
            return await pre_route

        # ── Intent gate: interpretation-first for question-like follow-ups ──
        # This reduces accidental tool calls when users ask about existing
        # calculations (especially after model fitting / sampling / ATE).
        intent = self._classify_user_intent(user_message)
        if intent in {"explain", "status_question"}:
            return await self._answer_general_question_with_interpretation_model(
                user_message=user_message,
                decision_model_draft=None,
            )
        if intent == "ambiguous" and self.state.current_step in {"model_fitted", "sampled", "ate_computed"}:
            return await self._answer_general_question_with_interpretation_model(
                user_message=user_message,
                decision_model_draft=None,
            )
        
        # ── AI Agent: LLM selects and executes tools ──
        try:
            return await self._agent_select_and_execute_tool(user_message)
        except Exception as e:
            print(f"[AGENT ERROR] Agent failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: try workflow-based routing
            print(f"[AGENT] Falling back to workflow routing")
            return await self._route_by_workflow_step(user_message)
    
    # =========================================================================
    # DETERMINISTIC PRE-ROUTING — catch workflow commands before the LLM
    # =========================================================================

    def _is_question_like_message(self, lower: str) -> bool:
        if "?" in lower:
            return True
        question_starts = (
            "what", "why", "how", "when", "where", "which",
            "can ", "could ", "would ", "should ", "is ", "are ",
            "do ", "does ", "did ", "am i ", "help me understand",
            "explain", "interpret", "compare",
        )
        return any(lower.startswith(s) for s in question_starts)

    def _is_explicit_tool_command(self, lower: str) -> bool:
        explicit_exact = {
            "fit model",
            "train model",
            "compute ate",
            "compute the ate",
            "calculate ate",
            "open dag editor",
            "dag editor",
            "interactive editor",
            "test dag",
            "download report",
            "generate report",
            "show all plots",
            "create plots",
            "use default",
            "use defaults",
            "default",
        }
        if lower in explicit_exact:
            return True

        action_match = re.match(
            r"^(please\s+)?(compute|calculate|run|fit|train|test|generate|create|open|download|show|sample)\b",
            lower,
        )
        if not action_match:
            return False

        objects = (
            "ate", "effect", "model", "dag", "editor", "report",
            "plot", "plots", "ci", "independence", "sample", "samples",
            "data point", "data points", "distribution",
        )
        return any(obj in lower for obj in objects)

    def _classify_user_intent(self, user_message: str) -> str:
        lower = (user_message or "").lower().strip()
        if not lower:
            return "ambiguous"

        if any(k in lower for k in ("current step", "where am i", "workflow status", "status")):
            return "status_question"

        if self._is_explicit_tool_command(lower):
            return "tool_command"

        if self._is_question_like_message(lower):
            return "explain"

        # Post-calculation follow-ups should default to interpretation unless
        # users issue an explicit action command.
        if self.state.current_step in {"model_fitted", "sampled", "ate_computed"}:
            explain_terms = (
                "mean", "std", "confidence", "important", "clinical",
                "meaning", "interpret", "explain", "compare", "difference",
                "treated", "control", "effect size", "reliable",
            )
            if any(term in lower for term in explain_terms):
                return "explain"

        return "ambiguous"
    
    def _try_deterministic_route(self, user_message: str):
        """
        Intercept commands that the chatbot itself suggested via action buttons
        and that have no matching tool in ToolRegistry.
        
        Returns an awaitable (coroutine) if the message should be handled
        deterministically, or None if the agent loop should handle it.
        
        Why this exists:
          The action buttons emit phrases like "apply revisions" or "suggest
          revisions".  These are multi-step orchestration actions (read CI
          results → call LLM → mutate DAG) that live in the workflow engine.
          There is no single ToolRegistry tool for them, so the LLM agent
          cannot dispatch them correctly.  Intercepting them here guarantees
          the right handler is called.
        """
        lower = user_message.lower().strip()
        step = self.state.current_step

        # ── "open dag editor" — always open editor directly ──
        # This avoids accidental rerouting into DAG proposal logic when the
        # LLM tool-selection step fails or picks a different tool.
        if (
            lower in {"open dag editor", "dag editor", "interactive editor"}
            or lower.startswith("open dag editor")
            or lower.startswith("open the dag editor")
            or lower.startswith("open interactive editor")
        ):
            print(f"[PRE-ROUTE] Intercepted 'open dag editor' at step '{step}'")
            return self._open_dag_editor_direct()

        if lower in {
            "open uploaded pairplot",
            "open data pairplot",
            "show uploaded pairplot",
            "show data pairplot",
        }:
            print(f"[PRE-ROUTE] Intercepted upload pairplot request at step '{step}'")
            return self._show_uploaded_pairplot()
        
        # ── "apply revisions" — update DAG based on earlier LLM suggestions ──
        if self._is_explicit_tool_command(lower) and "apply" in lower and "revision" in lower:
            print(f"[PRE-ROUTE] Intercepted 'apply revisions' at step '{step}'")
            return self._apply_revisions_to_dag()
        
        # ── "suggest revisions" — ask LLM to analyse CI failures ──
        if self._is_explicit_tool_command(lower) and "suggest" in lower and "revision" in lower:
            print(f"[PRE-ROUTE] Intercepted 'suggest revisions' at step '{step}'")
            return self._step_4_propose_revisions()
        
        # ── "proceed with fitting" — explicit action button from CI test results ──
        if lower.startswith("proceed with fitting"):
            print(f"[PRE-ROUTE] Intercepted 'proceed with fitting' at step '{step}'")
            self.state.current_step = "dag_finalized"
            self.save_session_state()
            return self._step_5_fit_model()

        # ── "fit model" — allow direct fitting without forcing CI pass ──
        if (
            (lower.startswith("fit model") or lower.startswith("train model"))
            and self.state.proposed_dag is not None
            and self.state.data_path is not None
        ):
            print(f"[PRE-ROUTE] Intercepted direct fit request at step '{step}'")
            self.state.current_step = "dag_finalized"
            self.save_session_state()
            return self._step_5_fit_model()

        # If user is in DAG-ready stage and replies with generic "default",
        # interpret as "run the default DAG consistency test".
        if (
            lower in {"default", "use default", "use defaults"}
            and step == "dag_proposed"
            and not self.state.pending_tool
        ):
            print(f"[PRE-ROUTE] Intercepted default at dag_proposed -> test DAG")
            return self._step_3_test_dag()

        # Explicit ATE entrypoint used by action buttons.
        if lower in {"compute the ate", "compute ate", "calculate ate"}:
            print(f"[PRE-ROUTE] Intercepted generic ATE request at step '{step}'")
            return self._start_guided_compute_ate()

        # Sampling intent should run the sampling tool directly (not full create_plots),
        # so users don't get the large multi-plot bundle unless explicitly requested.
        if self._is_explicit_tool_command(lower) and "sample" in lower and (
            "data point" in lower
            or "data points" in lower
            or "samples" in lower
            or re.search(r"\bsample\s+\d+", lower)
        ):
            if "all plots" not in lower and "generate all" not in lower:
                print(f"[PRE-ROUTE] Intercepted sampling request at step '{step}'")
                return self._run_sample_from_text(user_message)
        
        # No deterministic match — let the agent loop handle it
        return None

    async def _open_dag_editor_direct(self) -> str:
        """Open DAG editor deterministically and initialize an empty DAG if needed."""
        if self.state.proposed_dag is None:
            vars = []
            if self.state.data_df is not None and hasattr(self.state.data_df, "columns"):
                vars = list(self.state.data_df.columns)
            else:
                session = sessions.get(self.session_id, {})
                vars = session.get("data_info", {}).get("columns", []) or []

            if vars:
                n = len(vars)
                adj_matrix = np.zeros((n, n), dtype=int)
                self.state.proposed_dag = {
                    "adjacency_matrix": adj_matrix.tolist(),
                    "variables": vars,
                    "edges": [],
                    "llm_explanation": "Initialized empty DAG for manual editing."
                }
                self.state.current_step = "dag_proposed"
                self.save_session_state()

        result = await self._execute_tool("open_dag_editor", {})
        return await self._format_tool_results([("open_dag_editor", result)], "open dag editor")

    async def _show_uploaded_pairplot(self) -> str:
        """Show the upload-time pairplot in chat."""
        session = sessions.get(self.session_id, {})
        url = session.get("data_pairplot_url")
        if not url:
            return (
                "I cannot find an uploaded data pairplot for this session yet.\n\n"
                "Please upload data first, then I will generate and show it."
            )
        return (
            "**Uploaded Data Pairplot (full-size):**\n\n"
            f"![Uploaded Data Pairplot]({url})"
        )
    
    # =========================================================================
    # CORE AGENT LOOP — LLM function calling → tool execution → response
    # =========================================================================

    async def _start_guided_compute_ate(self) -> str:
        """
        Start ATE flow without preselecting X/Y so guided prompts ask the user
        for treatment and outcome variables first.
        """
        result = await self._execute_tool("compute_ate", {})
        return await self._format_tool_results([("compute_ate", result)], "compute the ate")

    async def _run_sample_from_text(self, user_message: str) -> str:
        """Run sample tool with parsed sample count from user text."""
        msg = (user_message or "").lower()
        n_samples = 1000
        m = re.search(r"(\d+)\s*(?:sample|samples|data point|data points)", msg)
        if m:
            try:
                n_samples = max(1, int(m.group(1)))
            except Exception:
                n_samples = 1000
        result = await self._execute_tool("sample", {"n_samples": n_samples})
        return await self._format_tool_results([("sample", result)], user_message)
    
    async def _agent_select_and_execute_tool(self, user_message: str) -> str:
        """
        The AI Agent loop:
          1. Build context (current state, available data, DAG, model)
          2. Send to LLM with tool schemas → LLM picks a tool via function calling
          3. Execute the selected tool locally
          4. Update workflow state based on what tool ran
          5. Format the result into a user-friendly response
        
        If the LLM doesn't select a tool (just responds with text), we return
        the text response directly.
        """
        # ── Step 1: Build context for the LLM ──
        dag_vars = []
        dag_edges = []
        if self.state.proposed_dag:
            dag_vars = self.state.proposed_dag.get("variables", [])
            dag_edges = self.state.proposed_dag.get("edges", [])
        
        data_vars = []
        if self.state.data_df is not None:
            data_vars = list(self.state.data_df.columns)
        
        context = {
            "current_step": self.state.current_step,
            "has_data": self.state.data_path is not None,
            "data_variables": data_vars,
            "has_dag": self.state.proposed_dag is not None,
            "dag_variables": dag_vars,
            "dag_edges": [[p, c] for p, c in dag_edges] if dag_edges else [],
            "has_fitted_model": self.state.fitted_model is not None,
            "experiment_dir": self.state.experiment_dir or "N/A",
            "has_ci_test_results": self.state.ci_test_results is not None,
        }
        
        if self.state.ci_test_results:
            context["ci_tests_consistent"] = self.state.ci_test_results.get("consistent", False)
            context["ci_tests_rejected_count"] = self.state.ci_test_results.get("rejected_count", 0)
        
        # ── Step 2: Call LLM with function calling ──
        system_prompt = self._build_agent_system_prompt()
        
        user_prompt = f"""CURRENT STATE:
{json.dumps(context, indent=2)}

USER MESSAGE: {user_message}

Based on the user's message and the current state, select the appropriate tool to call. 
If no tool is needed (e.g., the user is just asking a question you can answer), respond directly.
If the user says "yes" after a DAG is proposed, call test_dag.
If the user wants to fit the model, call fit_model.
Always fill in tool parameters from the current state (e.g., use dag_variables for variable names)."""
        
        tool_schemas = ToolRegistry.get_llm_tool_schemas()
        
        print(f"[AGENT] Sending to LLM with {len(tool_schemas)} tools available")
        
        llm_response = llm_client.chat.completions.create(
            model=llm_model_decision,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=tool_schemas,
            tool_choice="auto",
            temperature=0.1
        )
        
        message = llm_response.choices[0].message
        
        # ── Step 3: If LLM selected a tool, execute it ──
        if not message.tool_calls:
            # LLM chose no tool; answer with the interpretation model instead.
            print(f"[AGENT] LLM responded directly (no tool call)")
            return await self._answer_general_question_with_interpretation_model(
                user_message=user_message,
                decision_model_draft=message.content,
            )
        
        # Execute each tool call
        all_results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

            # Enforce guided fit setup: even if the LLM auto-fills hyperparameters,
            # route first through the interactive pending-parameter flow.
            if tool_name == "fit_model":
                for key in ("epochs", "learning_rate", "batch_size"):
                    tool_args.pop(key, None)
            
            print(f"[AGENT] LLM selected tool: {tool_name}")
            print(f"[AGENT] Tool arguments: {json.dumps(tool_args, indent=2)}")
            
            result = await self._execute_tool(tool_name, tool_args)
            all_results.append((tool_name, result))
        
        # ── Step 4 & 5: Format response (state is updated inside _execute_tool) ──
        return await self._format_tool_results(all_results, user_message)
    
    def _build_agent_system_prompt(self) -> str:
        """System prompt that tells the LLM what tools are available and how to use them."""
        return """You are a causal inference AI agent. You help users perform causal analysis
using the TRAM-DAG framework. You have access to statistical tools that run locally.

AVAILABLE TOOLS:
  1. propose_dag       — Propose a causal DAG from variable names
  2. test_dag          — Run conditional independence tests on the DAG
  3. fit_model         — Fit the TRAM-DAG model (takes a few minutes)
  4. sample            — Sample data from the fitted model
  5. compute_ate       — Compute Average Treatment Effect (causal effect)
  6. create_plots      — Generate visualizations (distributions, loss, DAG, intervention)
  7. show_associations — Show correlation matrix from the data
  8. generate_report   — Generate a PDF report with all results
  9. open_dag_editor   — Open the visual DAG editor in the UI

WORKFLOW (typical order):
  Data uploaded → propose_dag → test_dag → fit_model → compute_ate / sample / create_plots

RULES:
- When user says "yes" or "test" after a DAG is proposed → call test_dag
- When user says "fit model" or "proceed" → call fit_model
- When user explicitly asks to run a new effect/treatment computation → call compute_ate
- When user asks for plots or visualizations → call create_plots
- When user asks about correlations or associations → call show_associations
- When user asks for a report → call generate_report
- When user wants to edit the DAG visually → call open_dag_editor
- If the user asks to explain, compare, interpret, or clarify existing results,
  do NOT call tools; answer directly in plain language.
- You do NOT need to fill in parameters that come from state (dag, data_path, experiment_dir)
  — the execution layer handles those automatically.
- For propose_dag, pass the variable names from data_variables in the state.
- For compute_ate, pass X (treatment) and Y (outcome) variable names.
- For create_plots, specify which plot_types the user wants.
- If user's intent is unclear, respond with helpful text (no tool call)."""
    
    async def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """
        Execute a tool selected by the AI Agent.
        
        This method:
          1. Fills in missing parameters from session state
          2. Calls the corresponding CausalTools method
          3. Updates workflow state based on the tool that ran
          4. Returns the result dict
        """
        result = {"tool": tool_name, "success": False, "data": None, "error": None, "plots": []}
        
        try:
            # Preflight checks that should fail fast (before guided param prompts).
            if tool_name == "fit_model" and (not self.state.proposed_dag or not self.state.data_path):
                result["error"] = "DAG or data not available."
                return result

            guided_prompt = self._maybe_prompt_missing_params(tool_name, tool_args)
            if guided_prompt is not None:
                return guided_prompt

            # ── TOOL: propose_dag ──
            if tool_name == "propose_dag":
                # Always prefer real dataset columns over LLM-supplied vars.
                # This prevents hallucinated semantic variable names from
                # bypassing the "generic variable names" safeguard.
                vars_list = []
                if self.state.data_df is not None:
                    vars_list = list(self.state.data_df.columns)
                elif self.session_id in sessions:
                    vars_list = sessions[self.session_id].get("data_info", {}).get("columns", []) or []
                if not vars_list:
                    vars_list = tool_args.get("vars", [])
                
                # Check if variable names are meaningful enough for the LLM
                if not self._are_variable_names_meaningful(vars_list):
                    # Generic names (x1, x2, col1, …) → cannot infer DAG
                    n = len(vars_list)
                    dag_result = {
                        "adjacency_matrix": np.zeros((n, n), dtype=int).tolist(),
                        "variables": vars_list,
                        "edges": [],
                        "llm_explanation": (
                            "Cannot infer a DAG — variable names are generic and "
                            "do not carry enough semantic meaning."
                        ),
                    }
                    self.state.proposed_dag = dag_result
                    self._invalidate_runtime_after_dag_change()
                    self.state.current_step = "dag_proposed"
                    self.save_session_state()
                    result["success"] = True
                    result["data"] = {
                        "variables": vars_list,
                        "edges": [],
                        "explanation": dag_result["llm_explanation"],
                        "_generic_names": True,   # marker for _format_tool_results
                    }
                    return result
                
                dag_result = CausalTools.propose_dag_from_llm(
                    vars=vars_list,
                    expert_text=tool_args.get("expert_text")
                )
                self.state.proposed_dag = dag_result
                self._invalidate_runtime_after_dag_change()
                self.state.current_step = "dag_proposed"
                self.save_session_state()
                
                # Generate DAG plot
                try:
                    plot_result = CausalTools._create_dag_plot_with_ci(self.session_id, dag_result)
                    if plot_result:
                        img_path, implied_cis = plot_result
                        plot_filename = Path(img_path).name
                        result["plots"].append(f"/api/plots/{plot_filename}")
                        result["implied_cis"] = [
                            f"{ci['x']} \u27C2 {ci['y']}" + (f" | {', '.join(ci['conditioning_set'])}" if ci['conditioning_set'] else "")
                            for ci in implied_cis
                        ]
                except Exception as e:
                    print(f"[AGENT] Failed to generate DAG plot: {e}")
                
                result["success"] = True
                result["data"] = {
                    "variables": dag_result.get("variables", []),
                    "edges": dag_result.get("edges", []),
                    "explanation": dag_result.get("llm_explanation", "")
                }
                result["plots"].extend(
                    CausalTools.import_mcp_artifacts(
                        self.session_id, dag_result.get("mcp_artifacts", [])
                    )
                )
            
            # ── TOOL: test_dag ──
            elif tool_name == "test_dag":
                if not self.state.proposed_dag or not self.state.data_path:
                    result["error"] = "DAG or data not available. Please propose a DAG and upload data first."
                    return result
                
                alpha = tool_args.get("alpha", 0.05)
                ci_results = CausalTools.test_dag_consistency(
                    dag=self.state.proposed_dag,
                    data_path=self.state.data_path,
                    alpha=alpha
                )
                self.state.ci_test_results = ci_results
                self.state.current_step = "dag_tested"
                self.save_session_state()
                
                # Generate CI results plot
                try:
                    plot_path = CausalTools._create_dag_plot_with_ci_results(
                        self.session_id, self.state.proposed_dag, ci_results
                    )
                    if plot_path and Path(plot_path).exists():
                        result["plots"].append(f"/api/plots/{Path(plot_path).name}")
                except Exception as e:
                    print(f"[AGENT] Failed to generate CI results plot: {e}")

                # If MCP returned pre-rendered plots from DAG_Validator_Agent,
                # import them as well.
                result["plots"].extend(
                    CausalTools.import_mcp_artifacts(
                        self.session_id, ci_results.get("mcp_artifacts", [])
                    )
                )
                
                result["success"] = True
                result["data"] = ci_results
            
            # ── TOOL: fit_model ──
            elif tool_name == "fit_model":
                # Source of truth: refresh DAG from shared session state so
                # manual editor saves are always picked up before fitting.
                latest_session = sessions.get(self.session_id, {})
                latest_dag = latest_session.get("proposed_dag")
                if latest_dag:
                    self.state.proposed_dag = latest_dag

                if not self.state.proposed_dag or not self.state.data_path:
                    result["error"] = "DAG or data not available."
                    return result
                
                # Normalize DAG payload to avoid malformed adjacency structures.
                dag_payload = dict(self.state.proposed_dag)
                vars_list = list(dag_payload.get("variables", []) or [])
                raw_edges = dag_payload.get("edges", []) or []
                normalized_edges: List[List[str]] = []
                for e in raw_edges:
                    if isinstance(e, (list, tuple)) and len(e) == 2:
                        p, c = str(e[0]), str(e[1])
                        if p and c:
                            normalized_edges.append([p, c])
                dag_payload["edges"] = normalized_edges
                n_vars = len(vars_list)
                adj = dag_payload.get("adjacency_matrix")
                valid_adj = (
                    isinstance(adj, list)
                    and len(adj) == n_vars
                    and all(isinstance(row, list) and len(row) == n_vars for row in adj)
                )
                if not valid_adj and n_vars > 0:
                    idx = {v: i for i, v in enumerate(vars_list)}
                    rebuilt = [[0 for _ in range(n_vars)] for _ in range(n_vars)]
                    for p, c in normalized_edges:
                        if p in idx and c in idx and p != c:
                            rebuilt[idx[p]][idx[c]] = 1
                    dag_payload["adjacency_matrix"] = rebuilt
                
                run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                downloads_root = EXPERIMENTS_BASE_DIR
                experiment_dir = str(downloads_root / self.session_id / f"fit_{run_id}")
                Path(experiment_dir).mkdir(parents=True, exist_ok=True)
                
                fit_result = CausalTools.fit_tramdag_model(
                    dag=dag_payload,
                    data_path=self.state.data_path,
                    experiment_dir=experiment_dir,
                    epochs=tool_args.get("epochs", 100),
                    learning_rate=tool_args.get("learning_rate", 0.01),
                    batch_size=tool_args.get("batch_size", 512),
                    random_seed=tool_args.get("random_seed", 42),
                )
                resolved_experiment_dir = fit_result.get("experiment_dir", experiment_dir)
                self.state.experiment_dir = resolved_experiment_dir
                self.state.fitted_model = fit_result
                self.state.current_step = "model_fitted"
                self.save_session_state()

                # Artifact-first: import canonical TRAM plots from MCP wrapper.
                fit_artifact_urls = self._import_and_store_mcp_artifacts(
                    fit_result.get("mcp_artifacts", []),
                    fit_result.get("artifact_manifest", []),
                )
                result["plots"].extend(fit_artifact_urls)

                # Enforce TRAM workflow parity: do not auto-render legacy local plots here.
                if not fit_artifact_urls:
                    result["data"] = {
                        "experiment_dir": resolved_experiment_dir,
                        "status": "fitted",
                        "artifact_manifest": fit_result.get("artifact_manifest", []),
                        "plot_warning": "No canonical TRAM artifacts were returned by MCP wrapper.",
                    }
                    result["success"] = True
                    return result
                
                result["success"] = True
                result["data"] = {
                    "experiment_dir": resolved_experiment_dir,
                    "status": "fitted",
                    "artifact_manifest": fit_result.get("artifact_manifest", []),
                }
            
            # ── TOOL: sample ──
            elif tool_name == "sample":
                if not self.state.experiment_dir:
                    result["error"] = "Model not fitted yet. Please fit the model first."
                    return result
                
                sample_result = CausalTools.sample_from_model(
                    experiment_dir=self.state.experiment_dir,
                    n_samples=tool_args.get("n_samples", 10000),
                    do_interventions=tool_args.get("do_interventions", {}),
                    random_seed=tool_args.get("random_seed", 42),
                    include_metadata=True,
                )
                samples = sample_result.get("samples", {})
                
                # Summarize (don't send raw samples to LLM)
                summary = {}
                for node, data in samples.items():
                    values = data["values"]
                    summary[node] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values))
                    }

                sample_artifact_urls = self._import_and_store_mcp_artifacts(
                    sample_result.get("mcp_artifacts", []),
                    sample_result.get("artifact_manifest", []),
                )
                result["plots"].extend(sample_artifact_urls)
                
                result["success"] = True
                result["data"] = {
                    "n_samples": tool_args.get("n_samples", 10000),
                    "interventions": tool_args.get("do_interventions", {}),
                    "summary": summary,
                    "artifact_manifest": sample_result.get("artifact_manifest", []),
                }
                self.state.current_step = "sampled"
            
            # ── TOOL: compute_ate ──
            elif tool_name == "compute_ate":
                if not self.state.experiment_dir:
                    result["error"] = "Model not fitted yet. Please fit the model first."
                    return result
                
                X = tool_args.get("X")
                Y = tool_args.get("Y")
                if not X or not Y:
                    result["error"] = "Treatment variable (X) and outcome variable (Y) are required."
                    return result
                
                ate_result = CausalTools.compute_ate(
                    experiment_dir=self.state.experiment_dir,
                    X=X, Y=Y,
                    x_treated=tool_args.get("x_treated", 1.0),
                    x_control=tool_args.get("x_control", 0.0),
                    n_samples=tool_args.get("n_samples", 10000),
                    random_seed=tool_args.get("random_seed", 42),
                )
                ate_result["X"] = X
                ate_result["Y"] = Y
                
                # Save for report generation
                self._set_session_ate_result(ate_result)
                
                ate_artifact_urls = self._import_and_store_mcp_artifacts(
                    ate_result.get("mcp_artifacts", []),
                    ate_result.get("artifact_manifest", []),
                )
                result["plots"].extend(ate_artifact_urls)

                # Fallback local intervention plot only if MCP did not provide one.
                if not ate_artifact_urls:
                    try:
                        ate_plot = CausalTools._create_intervention_plot(self.state.experiment_dir, ate_result)
                        if ate_plot and Path(ate_plot).exists():
                            result["plots"].append(f"/api/plots/{Path(ate_plot).name}")
                    except Exception as e:
                        print(f"[AGENT] Failed to generate ATE plot: {e}")
                
                result["success"] = True
                result["data"] = ate_result
                self.state.current_step = "ate_computed"
            
            # ── TOOL: create_plots ──
            elif tool_name == "create_plots":
                if not self.state.experiment_dir:
                    result["error"] = "Model not fitted yet. Please fit the model first."
                    return result
                
                plot_types = tool_args.get("plot_types", ["all"])
                n_samples = tool_args.get("n_samples", 10000)
                gen_all = "all" in plot_types

                requested_kinds = set()
                if gen_all or "loss" in plot_types:
                    requested_kinds.update({"loss_history"})
                if gen_all or "dag" in plot_types:
                    requested_kinds.update({"dag_structure", "hdag"})
                if gen_all or "distributions" in plot_types:
                    requested_kinds.update({"sampling_distributions", "samples_vs_true"})
                if gen_all or "intervention" in plot_types:
                    requested_kinds.update({"intervention"})
                if gen_all or "linear_shift_history" in plot_types:
                    requested_kinds.update({"linear_shift_history"})
                if gen_all or "simple_intercepts_history" in plot_types:
                    requested_kinds.update({"simple_intercepts_history"})
                if gen_all or "latents" in plot_types:
                    requested_kinds.update({"latents"})
                if gen_all or "samples_vs_true" in plot_types:
                    requested_kinds.update({"samples_vs_true"})

                # First use stored MCP canonical plot URLs.
                for url in self._get_stored_mcp_plot_urls(list(requested_kinds)):
                    if url not in result["plots"]:
                        result["plots"].append(url)

                # If distribution-related plots are requested but not yet available,
                # trigger deterministic MCP sampling to produce canonical artifacts.
                need_distribution = bool({"sampling_distributions", "samples_vs_true"} & requested_kinds)
                have_distribution = any(
                    u in result["plots"]
                    for u in self._get_stored_mcp_plot_urls(["sampling_distributions", "samples_vs_true"])
                )
                if need_distribution and not have_distribution:
                    try:
                        sample_result = CausalTools.sample_from_model(
                            experiment_dir=self.state.experiment_dir,
                            n_samples=n_samples,
                            do_interventions={},
                            random_seed=tool_args.get("random_seed", 42),
                            include_metadata=True,
                        )
                        sample_urls = self._import_and_store_mcp_artifacts(
                            sample_result.get("mcp_artifacts", []),
                            sample_result.get("artifact_manifest", []),
                        )
                        for url in sample_urls:
                            if url not in result["plots"]:
                                result["plots"].append(url)
                    except Exception as e:
                        print(f"[AGENT] MCP distribution artifact generation failed: {e}")

                # Fallback local plot generation only for any still-missing essentials.
                if (gen_all or "distributions" in plot_types) and not any(
                    u in result["plots"]
                    for u in self._get_stored_mcp_plot_urls(["sampling_distributions", "samples_vs_true"])
                ):
                    try:
                        dist_path = CausalTools._create_distribution_plot(self.state.experiment_dir, self.session_id, n_samples=n_samples)
                        if dist_path and Path(dist_path).exists():
                            result["plots"].append(f"/api/plots/{Path(dist_path).name}")
                    except Exception as e:
                        print(f"[AGENT] Distribution plot failed: {e}")

                if (gen_all or "loss" in plot_types) and not self._get_stored_mcp_plot_urls(["loss_history"]):
                    try:
                        loss_path = CausalTools._create_loss_plot(self.state.experiment_dir, session_id=self.session_id)
                        if loss_path and Path(loss_path).exists():
                            result["plots"].append(f"/api/plots/{Path(loss_path).name}")
                    except Exception as e:
                        print(f"[AGENT] Loss plot failed: {e}")

                if (gen_all or "dag" in plot_types) and not self._get_stored_mcp_plot_urls(["dag_structure"]):
                    try:
                        if self.state.proposed_dag:
                            dag_path = CausalTools._create_dag_plot(self.session_id, self.state.proposed_dag)
                            if dag_path and Path(dag_path).exists():
                                result["plots"].append(f"/api/plots/{Path(dag_path).name}")
                    except Exception as e:
                        print(f"[AGENT] DAG plot failed: {e}")

                if (gen_all or "intervention" in plot_types) and not self._get_stored_mcp_plot_urls(["intervention"]):
                    try:
                        ate_data = sessions.get(self.session_id, {}).get("query_results", {}).get("ate")
                        if ate_data:
                            ate_plot = CausalTools._create_intervention_plot(self.state.experiment_dir, ate_data)
                            if ate_plot and Path(ate_plot).exists():
                                result["plots"].append(f"/api/plots/{Path(ate_plot).name}")
                    except Exception as e:
                        print(f"[AGENT] Intervention plot failed: {e}")
                
                result["success"] = True
                result["data"] = {"plot_types": plot_types, "num_plots": len(result["plots"])}
            
            # ── TOOL: show_associations ──
            elif tool_name == "show_associations":
                if self.state.data_df is None:
                    result["error"] = "No data uploaded yet."
                    return result
                
                corr = self.state.data_df.corr()
                result["success"] = True
                result["data"] = {
                    "correlation_matrix": corr.to_string(),
                    "variables": list(corr.columns)
                }
            
            # ── TOOL: generate_report ──
            elif tool_name == "generate_report":
                reports_dir = REPORTS_DIR
                reports_dir.mkdir(exist_ok=True)
                report_path = reports_dir / f"{self.session_id}_report.pdf"
                report_type = tool_args.get("report_type", "full")
                
                CausalTools.generate_report(self.session_id, str(report_path), report_type=report_type)
                
                self._set_session_report_path(report_path)
                
                result["success"] = True
                result["data"] = {"report_path": str(report_path), "report_type": report_type}
            
            # ── TOOL: open_dag_editor ──
            elif tool_name == "open_dag_editor":
                result["success"] = True
                result["data"] = {"action": "open_dag_editor"}
            
            else:
                result["error"] = f"Unknown tool: {tool_name}"
        
        except Exception as e:
            print(f"[AGENT] Tool {tool_name} failed: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
        
        return result
    
    async def _format_tool_results(self, tool_results: list, user_message: str) -> str:
        """
        Format tool execution results into a user-friendly chat response.
        
        This turns raw tool outputs into markdown with plots, tables,
        and action suggestions that the chat UI can render.
        """
        response_parts = []
        
        for tool_name, result in tool_results:
            if result.get("error"):
                response_parts.append(f"**Error:** {result['error']}")
                continue
            
            data = result.get("data", {})
            plots = result.get("plots", [])

            if isinstance(data, dict) and data.get("_guided_prompt"):
                response_parts.append(data["_guided_prompt"])
                guided_actions = data.get("_guided_actions", [])
                if guided_actions:
                    response_parts.append(self._format_action_suggestions(guided_actions))
                continue
            
            # ── Format: propose_dag ──
            if tool_name == "propose_dag":
                edges = data.get("edges", [])
                variables = data.get("variables", [])
                explanation = data.get("explanation", "")
                is_generic = data.get("_generic_names", False)
                
                if is_generic:
                    # ── Generic variable names: cannot infer DAG ──
                    response_parts.append("**Cannot automatically infer a DAG from your variable names.**\n")
                    response_parts.append(
                        f"Your dataset contains the variables: **{', '.join(variables)}**\n"
                    )
                    response_parts.append(
                        "These names are generic (e.g. `x1`, `x2`, `V1`, `col1`, …) and do not "
                        "carry enough semantic meaning for me to propose a causal structure.\n"
                    )
                    response_parts.append(
                        "**Please build your DAG manually** using the interactive DAG editor, "
                        "for all manual edge changes.\n"
                    )
                    
                    example_vars = variables[:2] if len(variables) >= 2 else variables
                    example_edge = (
                        f"{example_vars[0]} -> {example_vars[1]}"
                        if len(example_vars) == 2
                        else "var1 -> var2"
                    )
                    actions = [
                        {"label": "Use the interactive DAG editor", "command": "open dag editor",
                         "description": "Drag-and-drop interface to build your DAG visually"},
                        {"label": "Modify DAG in the editor", "command": "open dag editor",
                         "description": "Use the visual editor for all manual edge changes"},
                    ]
                    response_parts.append(self._format_action_suggestions(actions))
                else:
                    # ── Meaningful names: show proposed DAG ──
                    response_parts.append("**Stage 2: DAG proposal/edit**\n")
                    response_parts.append("**Proposed DAG Structure:**\n")
                    response_parts.append(f"**Variables:** {', '.join(variables)}\n")
                    response_parts.append("**Edges:**")
                    for parent, child in edges:
                        response_parts.append(f"- {parent} → {child}")
                    
                    if explanation:
                        response_parts.append(f"\n**Reasoning:** {self._clean_llm_response(explanation)}")
                    
                    # Show plot
                    for plot_url in plots:
                        sep = '?' if '?' not in plot_url else '&'
                        response_parts.append(f"\n![DAG]({plot_url}{sep}t={int(datetime.now().timestamp())})")
                    
                    # Show implied CIs if available
                    implied_cis = result.get("implied_cis", [])
                    if implied_cis:
                        response_parts.append(f"\n**Implied Conditional Independence Tests ({len(implied_cis)}):**")
                        response_parts.append("These are the testable implications of your DAG:\n")
                        for ci in implied_cis:
                            response_parts.append(f"- {ci}")
                    
                    # Action suggestions
                    actions = [
                        {"label": "Test this DAG against your data", "command": "yes, test the model",
                         "description": "Run conditional independence tests to validate the DAG"},
                        {"label": "Open DAG editor", "command": "open dag editor",
                         "description": "Visually edit the DAG structure"},
                        {"label": "Propose a different DAG", "command": "propose a different DAG",
                         "description": "Get an alternative causal structure"}
                    ]
                    response_parts.append(self._format_action_suggestions(actions))
            
            # ── Format: test_dag ──
            elif tool_name == "test_dag":
                response_parts.append("**Stage 3: DAG test and revision decision**\n")
                consistent = data.get("consistent", False)
                rejected = data.get("rejected_count", 0)
                tests = data.get("tests", [])
                total = len(tests)
                passed = total - rejected
                
                for plot_url in plots:
                    response_parts.append(f"![CI Test Results]({plot_url})")
                
                response_parts.append("\n**Consistency Test Results:**\n")
                response_parts.append(f"- **Consistent:** {'Yes' if consistent else 'No'}")
                response_parts.append(f"- **Total tests:** {total}")
                response_parts.append(f"- **Passed:** {passed}")
                response_parts.append(f"- **Rejected:** {rejected}")
                
                if tests:
                    response_parts.append("\n**Individual Tests:**\n")
                    for t in tests:
                        ci_str = t.get("ci", "?")
                        rej = t.get("rejected", False)
                        p_val = t.get("adj_p_value", t.get("p_value"))
                        test_method = (
                            t.get("test_name")
                            or t.get("test")
                            or t.get("method")
                            or t.get("ci_test")
                            or t.get("test_type")
                            or "CI"
                        )
                        status = "REJECTED" if rej else "Passed"
                        p_str = f"p={p_val:.4f}" if p_val is not None else ""
                        response_parts.append(
                            f"- `{test_method}` — {status} (rejected={str(bool(rej)).lower()}) — `{ci_str}` ({p_str})"
                        )
                
                if rejected > 0:
                    response_parts.append("\n**Some CI assumptions were rejected.** The DAG may need revision.\n")
                    actions = [
                        {"label": "Open DAG editor", "command": "open dag editor",
                         "description": "Manually fix the DAG using the visual editor"},
                        {"label": "Fit model anyway", "command": "fit model",
                         "description": "Proceed to model fitting despite inconsistencies"}
                    ]
                else:
                    response_parts.append("\n**The DAG is consistent with your data!** All CI tests passed.\n")
                    actions = [
                        {"label": "Fit the TRAM-DAG model", "command": "fit model",
                         "description": "Train the model using this DAG (takes a few minutes)"},
                        {"label": "Open DAG editor", "command": "open dag editor",
                         "description": "Make changes before fitting"}
                    ]
                response_parts.append(self._format_action_suggestions(actions))
            
            # ── Format: fit_model ──
            elif tool_name == "fit_model":
                response_parts.append("**Stage 4: Model fit completed**\n")
                response_parts.append("**Model fitted successfully!**\n")
                if data.get("plot_warning"):
                    response_parts.append(f"**Plot status:** {data['plot_warning']}\n")

                # Prefer TRAM-style canonical ordering from MCP artifact kinds.
                kind_label_order = [
                    ("loss_history", "Training Loss History"),
                    ("samples_vs_true", "Samples vs True"),
                    ("dag_structure", "DAG Structure"),
                    ("linear_shift_history", "Linear Shift History"),
                    ("simple_intercepts_history", "Simple Intercepts History"),
                    ("hdag", "h-DAG"),
                    ("latents", "Latent Distributions"),
                    ("sampling_distributions", "Sampling Distributions"),
                    ("intervention", "Intervention Plot"),
                ]
                stored = sessions.get(self.session_id, {}).get("mcp_plot_urls_by_kind", {})
                used_urls = set()
                for kind, label in kind_label_order:
                    url = stored.get(kind)
                    if url:
                        used_urls.add(url)
                        response_parts.append(f"**{label}:**")
                        response_parts.append(f"![{label}]({url})\n")

                # Include any remaining plots that were returned but not mapped.
                for plot_url in plots:
                    if plot_url in used_urls:
                        continue
                    response_parts.append("**Plot:**")
                    response_parts.append(f"![Plot]({plot_url})\n")
                
                dag_vars = self.state.proposed_dag.get("variables", []) if self.state.proposed_dag else []
                if not dag_vars:
                    dag_vars = self._example_vars()
                example_var = dag_vars[0]
                example_outcome = dag_vars[-1] if len(dag_vars) > 1 else dag_vars[0]
                
                actions = [
                    {"label": "Sample and plot", "command": "sample 1000 data points and show me the plots",
                     "description": "Generate distribution and loss plots"},
                    {"label": "Compute treatment effect", "command": "compute the ate",
                     "description": "Choose treatment (X) and outcome (Y), then compute ATE"},
                    {"label": "Download report", "command": "generate report",
                     "description": "Get a comprehensive PDF report"}
                ]
                response_parts.append(self._format_action_suggestions(actions))
                response_parts.append(self._calculation_followup_note())
            
            # ── Format: sample ──
            elif tool_name == "sample":
                interventions = data.get("interventions", {})
                summary = data.get("summary", {})
                
                if interventions:
                    interv_str = ", ".join(f"{k}={v}" for k, v in interventions.items())
                    response_parts.append(f"**Intervention Results: do({interv_str})**\n")
                else:
                    response_parts.append(f"**Sampling Results ({data.get('n_samples', '?')} samples):**\n")

                # Show TRAM-style model-vs-test diagnostics first when available.
                stored = sessions.get(self.session_id, {}).get("mcp_plot_urls_by_kind", {})
                samples_vs_true_url = stored.get("samples_vs_true")
                sampling_dist_url = stored.get("sampling_distributions")
                shown = set()
                if samples_vs_true_url:
                    response_parts.append("**Model Fit Check (Test Data vs Sampled):**")
                    response_parts.append(f"![Samples vs True]({samples_vs_true_url})\n")
                    shown.add(samples_vs_true_url)
                if sampling_dist_url:
                    response_parts.append("**Sampling Distributions:**")
                    response_parts.append(f"![Sampling Distributions]({sampling_dist_url})\n")
                    shown.add(sampling_dist_url)
                for plot_url in plots:
                    if plot_url in shown:
                        continue
                    response_parts.append(f"![Sample Plot]({plot_url})\n")
                
                for var, stats in summary.items():
                    response_parts.append(f"- **{var}**: mean={stats['mean']:.4f}, std={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}]")

                # Keep guided momentum: after sampling, steer user to ATE next.
                response_parts.append("\n**Next recommended step:** compute ATE with your chosen treatment and outcome variables.\n")
                actions = [
                    {
                        "label": "Compute treatment effect (ATE)",
                        "command": "compute the ate",
                        "description": "I'll guide you to pick treatment (X), outcome (Y), and intervention values"
                    },
                    {
                        "label": "Sample again with a different size",
                        "command": "sample 5000 data points",
                        "description": "Run another sampling quality check"
                    },
                    {
                        "label": "Download report",
                        "command": "generate report",
                        "description": "Create a PDF with your current analysis results"
                    },
                ]
                response_parts.append(self._format_action_suggestions(actions))
                response_parts.append(self._calculation_followup_note())
            
            # ── Format: compute_ate ──
            elif tool_name == "compute_ate":
                X = data.get("X", "?")
                Y = data.get("Y", "?")
                ate = data.get("ate", 0)
                x_treated = data.get("x_treated", 1.0)
                x_control = data.get("x_control", 0.0)
                
                response_parts.append("**Average Treatment Effect (ATE)**\n")
                response_parts.append(f"- **Treatment:** do({X} = {x_treated})")
                response_parts.append(f"- **Control:** do({X} = {x_control})")
                response_parts.append(f"- **Outcome:** {Y}")
                response_parts.append(f"- **ATE = {ate:.4f}**")
                response_parts.append(f"- E[{Y} | do({X}={x_treated})] = {data.get('y_treated_mean', 0):.4f} (std: {data.get('y_treated_std', 0):.4f})")
                response_parts.append(f"- E[{Y} | do({X}={x_control})] = {data.get('y_control_mean', 0):.4f} (std: {data.get('y_control_std', 0):.4f})")
                
                for plot_url in plots:
                    response_parts.append(f"\n![ATE Plot]({plot_url})\n")
                
                # Generate explanation
                try:
                    explanation = await self._generate_medical_explanation("ate", data)
                    if not explanation:
                        explanation = self._default_ate_interpretation(data)
                    response_parts.append(f"\n**What This Means:**\n\n{explanation}")
                except Exception:
                    response_parts.append(f"\n**What This Means:**\n\n{self._default_ate_interpretation(data)}")
                
                dag_vars = self.state.proposed_dag.get("variables", []) if self.state.proposed_dag else []
                actions = [
                    {"label": "Download report", "command": "download report",
                     "description": "Get PDF with ATE results and plots"},
                    {"label": "Generate plots", "command": "show me all plots",
                     "description": "Visualize distributions and loss history"},
                ]
                response_parts.append(self._format_action_suggestions(actions))
                response_parts.append(self._calculation_followup_note())
            
            # ── Format: create_plots ──
            elif tool_name == "create_plots":
                if plots:
                    response_parts.append(f"**Generated {len(plots)} plot(s):**\n")
                    for plot_url in plots:
                        response_parts.append(f"![Plot]({plot_url})\n")
                else:
                    response_parts.append("No plots could be generated.")
            
            # ── Format: show_associations ──
            elif tool_name == "show_associations":
                response_parts.append("**Correlation Matrix:**\n")
                response_parts.append("```\n" + data.get("correlation_matrix", "") + "\n```\n")
                response_parts.append("*Values close to 1 or -1 = strong association. Association ≠ causation!*")
            
            # ── Format: generate_report ──
            elif tool_name == "generate_report":
                response_parts.append("**PDF Report Generated!**\n")
                response_parts.append("Click the **Download Report** button in the header to download it.")
                if self.state.experiment_dir:
                    response_parts.append(
                        f"- Reproducible analysis scripts: `{Path(self.state.experiment_dir) / 'scripts'}`"
                    )
                    response_parts.append(
                        f"- Portable reproducible package: `{Path(self.state.experiment_dir) / 'reproducible_package'}`"
                    )
            
            # ── Format: open_dag_editor ──
            elif tool_name == "open_dag_editor":
                response_parts.append("<!-- OPEN_DAG_EDITOR -->")
                response_parts.append("**Opening the interactive DAG editor...**\n")
                response_parts.append("- **Click a source node**, then **click a target node** to add an edge")
                response_parts.append("- **Click on an edge** to remove it")
                response_parts.append("- Press **Save DAG** or **Save + Re-test** when done")
        
        return "\n".join(response_parts)
    
    # =========================================================================
    # FALLBACK — Workflow-based routing (used if agent fails)
    # =========================================================================
    
    async def _route_by_workflow_step(self, user_message: str) -> str:
        """Route message based on workflow step (existing logic)"""
        # CRITICAL: Double-check step from session right before routing
        session = sessions.get(self.session_id, {})
        session_step = session.get('current_step')
        if session_step and session_step != self.state.current_step:
            print(f"[CRITICAL] Step mismatch! Session has '{session_step}', state has '{self.state.current_step}'")
            print(f"[CRITICAL] Forcing state to match session: '{session_step}'")
            self.state.current_step = session_step
        
        print(f"[DEBUG] Routing to step based on current_step='{self.state.current_step}'")
        print(f"[DEBUG] Session current_step: {session.get('current_step')}")
        if self.state.current_step == "initial":
            print(f"[DEBUG] Routing to _step_1_parse_question")
            return await self._step_1_parse_question(user_message)
        elif self.state.current_step == "dag_proposed":
            print(f"[DEBUG] Routing to _step_2_handle_dag_response")
            return await self._step_2_handle_dag_response(user_message)
        elif self.state.current_step == "dag_tested":
            user_lower = user_message.lower().strip()
            
            # Check if user wants to open the DAG editor
            if "open dag editor" in user_lower or "dag editor" in user_lower or "interactive editor" in user_lower:
                print(f"[DEBUG] dag_tested: User wants to open DAG editor, routing back to step 2")
                self.state.current_step = "dag_proposed"
                self.save_session_state()
                return await self._step_2_handle_dag_response(user_message)
            
            # Check if user wants to modify the DAG with edge commands (e.g., "add x1 -> x3")
            edge_pattern = r'(?:add\s+)?([a-zA-Z0-9_]+)\s*[-=]>\s*([a-zA-Z0-9_]+)'
            if re.search(edge_pattern, user_message):
                print(f"[DEBUG] dag_tested: User wants to modify DAG edges, routing back to step 2")
                self.state.current_step = "dag_proposed"
                self.save_session_state()
                return await self._step_2_handle_dag_response(user_message)
            
            # Check if user wants to fit model
            wants_to_fit = ("fit" in user_lower or
                           ("proceed" in user_lower and "fitting" in user_lower) or
                           ("continue" in user_lower and "model" in user_lower) or
                           "proceed with fitting" in user_lower)
            
            if wants_to_fit:
                # User wants to fit - transition to finalized and fit
                print(f"[DEBUG] User wants to fit model, transitioning to 'dag_finalized' and fitting")
                self.state.current_step = "dag_finalized"
                self.save_session_state()
                return await self._step_5_fit_model()
            
            # Check if user wants to re-test the DAG
            if "test" in user_lower and "dag" in user_lower:
                print(f"[DEBUG] dag_tested: User wants to re-test DAG")
                return await self._step_3_test_dag()
            
            # Check if user wants to suggest or apply revisions
            if "suggest" in user_lower and "revision" in user_lower:
                print(f"[DEBUG] dag_tested: User wants revision suggestions")
                return await self._step_4_propose_revisions()
            
            if "apply" in user_lower and "revision" in user_lower:
                print(f"[DEBUG] dag_tested: User wants to apply revisions")
                return await self._apply_revisions_to_dag()
            
            # Default: show revisions or consistent message
            return await self._step_4_propose_revisions()
        elif self.state.current_step == "dag_finalized":
            user_lower = user_message.lower().strip()
            
            # Check if user wants to open the DAG editor instead of fitting
            if "open dag editor" in user_lower or "dag editor" in user_lower or "interactive editor" in user_lower:
                print(f"[DEBUG] dag_finalized: User wants to open DAG editor, routing back to step 2")
                self.state.current_step = "dag_proposed"
                self.save_session_state()
                return await self._step_2_handle_dag_response(user_message)
            
            # Check if user wants to modify edges
            edge_pattern = r'(?:add\s+)?([a-zA-Z0-9_]+)\s*[-=]>\s*([a-zA-Z0-9_]+)'
            if re.search(edge_pattern, user_message):
                print(f"[DEBUG] dag_finalized: User wants to modify DAG edges, routing back to step 2")
                self.state.current_step = "dag_proposed"
                self.save_session_state()
                return await self._step_2_handle_dag_response(user_message)
            
            # Check if user wants to re-test the DAG
            if "test" in user_lower and ("dag" in user_lower or "model" in user_lower):
                print(f"[DEBUG] dag_finalized: User wants to re-test DAG")
                self.state.current_step = "dag_proposed"
                self.save_session_state()
                return await self._step_3_test_dag()
            
            # Check if user wants to apply revisions
            if "apply" in user_lower and "revision" in user_lower:
                print(f"[DEBUG] dag_finalized: User wants to apply revisions")
                return await self._apply_revisions_to_dag()
            
            # Default: proceed to fit model
            return await self._step_5_fit_model()
        elif self.state.current_step in {"model_fitted", "sampled", "ate_computed"}:
            try:
                # Keep model-fitted behavior on the unified agent/tool path to
                # avoid legacy local plotting branches.
                result = await self._agent_select_and_execute_tool(user_message)
                if not result or result.strip() == "":
                    return self._get_fallback_response_for_query(user_message)
                return result
            except Exception as e:
                print(f"[ERROR] Error in _step_6_answer_query: {e}")
                import traceback
                traceback.print_exc()
                return self._get_fallback_response_for_query(user_message)
        else:
            # Unknown step - provide helpful guidance
            response = "**I'm ready to help with your causal inference question!**\n\n"
            response += "**Current status:** I'm in an unexpected state. Let me help you get started:\n\n"
            
            # Check what we have
            if self.state.data_path:
                response += "Data is uploaded\n"
            else:
                response += "No data uploaded yet\n"
            
            if self.state.proposed_dag:
                response += "DAG is proposed\n"
            else:
                response += "No DAG yet\n"
            
            if self.state.fitted_model:
                response += "Model is fitted\n"
            else:
                response += "Model not fitted yet\n"
            
            response += "\n**What would you like to do?**\n\n"
            
            actions = []
            if not self.state.data_path:
                actions.append({
                    "label": "Upload data",
                    "command": "Click the 'Upload Data' button above",
                    "description": "Start by uploading your CSV or Excel file"
                })
            elif not self.state.proposed_dag:
                actions.append({
                    "label": "Create a DAG",
                    "command": "create a dag",
                    "description": "Propose a causal structure for your data"
                })
            elif not self.state.fitted_model:
                actions.append({
                    "label": "Fit the model",
                    "command": "fit model",
                    "description": "Train the TRAM-DAG model"
                })
            else:
                dag_vars = self.state.proposed_dag.get("variables", [])
                if not dag_vars:
                    dag_vars = self._example_vars()
                example_var = dag_vars[0] if dag_vars else "x1"
                example_outcome = dag_vars[-1] if len(dag_vars) > 1 else example_var
                actions.append({
                    "label": "Ask a causal question",
                    "command": "compute the ate",
                    "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                })
            
            response += self._format_action_suggestions(actions)
            return response
    
    def _are_variable_names_meaningful(self, var_names: List[str]) -> bool:
        """
        Check whether variable names carry enough semantic meaning for an LLM
        to propose a plausible causal DAG.
        
        Returns True if the names are descriptive (e.g. 'age', 'income', 'blood_pressure').
        Returns False if ALL names match generic patterns like x1, x2, V1, col1, var_1, etc.
        
        The heuristic: if *any* variable name looks descriptive, we consider the
        set meaningful (the LLM can still try).  Only when *every* name is generic
        do we fall back to manual DAG building.
        """
        import re as _re
        
        # Patterns that indicate a generic / non-informative variable name
        generic_patterns = [
            r'^[xXyYzZvV]\d+$',          # x1, X2, y1, V3, z10
            r'^[xXyYzZvV]_\d+$',         # x_1, X_2, v_3
            r'^var[_]?\d+$',              # var1, var_1, var2
            r'^col[_]?\d+$',             # col1, col_1, col2
            r'^feature[_]?\d+$',          # feature1, feature_1
            r'^[a-zA-Z]$',               # single letter: a, b, X, Y
            r'^\d+$',                     # pure numbers: 0, 1, 2
            r'^Unnamed[:\s_]*\d*$',       # Unnamed: 0  (pandas default)
            r'^c\d+$',                    # c1, c2  (common generic)
            r'^f\d+$',                    # f1, f2
        ]
        
        compiled = [_re.compile(p, _re.IGNORECASE) for p in generic_patterns]
        
        for name in var_names:
            name_stripped = name.strip()
            is_generic = any(pat.match(name_stripped) for pat in compiled)
            if not is_generic:
                # At least one variable looks descriptive — treat the set as meaningful
                return True
        
        # Every single variable name matched a generic pattern
        return False
    
    async def _step_1_parse_question(self, question: str) -> str:
        """Step 1: Parse causal question and propose DAG"""
        self.state.query_type = "intervention"  # Default, could be improved with LLM
        
        # Get session data (refresh in case it was updated)
        session = sessions.get(self.session_id, {})
        
        # Load data from session if not already in state
        if self.state.data_df is None and session.get("data_df"):
            self.state.data_df = pd.DataFrame(session["data_df"])
        
        if self.state.data_path is None and session.get("data_path"):
            self.state.data_path = session["data_path"]
        
        # Extract variables from uploaded data
        vars = []
        if self.state.data_df is not None and hasattr(self.state.data_df, 'columns'):
            vars = list(self.state.data_df.columns)
        elif session.get("data_info", {}).get("columns"):
            vars = session["data_info"]["columns"]
        
        # Debug: log what we found
        print(f"[DEBUG] Session ID: {self.session_id}")
        print(f"[DEBUG] Has data_df in state: {self.state.data_df is not None}")
        print(f"[DEBUG] Has data_path in state: {self.state.data_path is not None}")
        print(f"[DEBUG] Has data_df in session: {session.get('data_df') is not None}")
        print(f"[DEBUG] Has data_info in session: {session.get('data_info') is not None}")
        print(f"[DEBUG] Variables found: {vars}")
        
        if not vars:
            response = "**No data found**\n\n"
            response += "To get started, you need to upload your data first.\n\n"
            
            actions = [
                {
                    "label": "Upload data file",
                    "command": "Click the 'Upload Data' button above",
                    "description": "Upload a CSV or Excel file with your data"
                },
                {
                    "label": "Specify variables in your question",
                    "command": "I have variables age, bmi, blood_pressure. Compute the ate.",
                    "description": "Tell me your variable names and I'll ask which X/Y to use"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response
        
        # Check if variable names are meaningful enough for LLM to infer a DAG
        meaningful = self._are_variable_names_meaningful(vars)
        
        if meaningful:
            # Variable names are descriptive — let the LLM propose a DAG
            response = "I'll analyze your question and propose an initial DAG structure.\n\n"
            
            dag_result = self.tools.propose_dag_from_llm(vars)
            self.state.proposed_dag = dag_result
            
            response += f"**Proposed DAG Structure:**\n\n"
            response += f"**Variables:** {', '.join(vars)}\n\n"
            response += f"**Edges:**\n"
            for parent, child in dag_result['edges']:
                response += f"- {parent} → {child}\n"
            
            response += "\nYou can also use the **interactive DAG editor** below to visually add or remove edges.\n"
            
            # Add action suggestions
            actions = [
                {
                    "label": "Test this DAG against your data",
                    "command": "yes, test the model",
                    "description": "Run conditional independence tests to check if the DAG is consistent with your data"
                },
                {
                    "label": "Modify the DAG structure",
                    "command": "open dag editor",
                    "description": "Use the visual DAG editor for manual edge changes"
                },
                {
                    "label": "Ask for a different DAG",
                    "command": "no, propose a different DAG",
                    "description": "I'll propose an alternative causal structure"
                }
            ]
            response += self._format_action_suggestions(actions)
        else:
            # Variable names are generic — cannot infer a meaningful DAG
            response = "**Cannot automatically infer a DAG from your variable names.**\n\n"
            response += (
                f"Your dataset contains the variables: **{', '.join(vars)}**\n\n"
                "These names are generic (e.g. `x1`, `x2`, `V1`, `col1`, ...) and do not carry "
                "enough semantic meaning for me to propose a causal structure.\n\n"
                "**Please build your DAG manually** using the interactive DAG editor below, "
                "for all manual edge changes.\n"
            )
            
            # Create an empty DAG (no edges) with the variables so the editor can be populated
            n = len(vars)
            adj_matrix = np.zeros((n, n), dtype=int)
            dag_result = {
                "adjacency_matrix": adj_matrix.tolist(),
                "variables": vars,
                "edges": [],
                "llm_explanation": "No DAG proposed — variable names are not semantically meaningful."
            }
            self.state.proposed_dag = dag_result
            
            # Add action suggestions for manual DAG building
            example_vars = vars[:2] if len(vars) >= 2 else vars
            example_edge = f"{example_vars[0]} -> {example_vars[1]}" if len(example_vars) == 2 else "var1 -> var2"
            actions = [
                {
                    "label": "Use the interactive DAG editor",
                    "command": "open dag editor",
                    "description": "Drag-and-drop interface to build your DAG visually"
                },
                {
                    "label": "Edit DAG structure",
                    "command": "open dag editor",
                    "description": "Open the editor and add/remove edges visually"
                }
            ]
            response += self._format_action_suggestions(actions)
        
        self.state.current_step = "dag_proposed"
        # Update session immediately - CRITICAL for state persistence
        if self.session_id in sessions:
            sessions[self.session_id]["current_step"] = "dag_proposed"
            sessions[self.session_id]["proposed_dag"] = self.state.proposed_dag
            print(f"[DEBUG] Set current_step to 'dag_proposed' and saved to session {self.session_id}")
            print(f"[DEBUG] Verified session current_step after save: {sessions[self.session_id].get('current_step')}")
        else:
            print(f"[ERROR] Session {self.session_id} not found when trying to save step!")
        return response
    
    async def _step_2_handle_dag_response(self, user_message: str) -> str:
        """Step 2: Handle user response to proposed DAG"""
        print(f"[DEBUG] _step_2_handle_dag_response called with message: '{user_message}'")
        print(f"[DEBUG] Current proposed_dag: {self.state.proposed_dag}")
        user_lower = user_message.lower().strip()
        
        # Check if user wants to open the interactive DAG editor
        if "open dag editor" in user_lower or "dag editor" in user_lower or "interactive editor" in user_lower:
            # Return a special marker that the frontend will detect to open the DAG editor modal
            response = "<!-- OPEN_DAG_EDITOR -->"
            response += "**Opening the interactive DAG editor...**\n\n"
            response += "Use the visual editor to build your DAG:\n"
            response += "- **Click a source node**, then **click a target node** to add a directed edge\n"
            response += "- **Click on an edge** to remove it\n"
            response += "- Press **Save DAG** when you're done\n"
            return response
        
        # Check if user wants to modify the DAG
        # Look for edge patterns like "x1 -> x2" or "x1->x2" or "add x1 -> x2"
        
        # Pattern to match edges: "var1 -> var2" or "var1->var2" or "add var1 -> var2"
        edge_pattern = r'(?:add\s+)?([a-zA-Z0-9_]+)\s*[-=]>\s*([a-zA-Z0-9_]+)'
        edges_found = re.findall(edge_pattern, user_message)
        
        # Check if user wants a completely different DAG proposal
        if "propose" in user_lower and ("different" in user_lower or "another" in user_lower or "new" in user_lower):
            print(f"[DEBUG] User wants a different DAG, re-running step 1")
            self.state.current_step = "initial"
            self.save_session_state()
            return await self._step_1_parse_question("Please propose a different DAG structure for my data")
        
        if edges_found or "no" in user_lower or "change" in user_lower or "modify" in user_lower:
            response = "<!-- OPEN_DAG_EDITOR -->"
            response += "**Manual DAG changes use the interactive editor.**\n\n"
            response += "I opened the DAG editor so you can modify edges visually:\n"
            response += "- **Click a source node**, then **click a target node** to add an edge\n"
            response += "- **Click an edge** to remove it\n"
            response += "- Press **Save DAG** when done\n"
            return response
        
        # Check if user wants to proceed with testing
        elif "yes" in user_lower or "test" in user_lower or "proceed" in user_lower or "continue" in user_lower:
            # User wants to test the DAG
            print(f"[DEBUG] User wants to test DAG, proceeding to step 3")
            return await self._step_3_test_dag()
        elif "fit" in user_lower and "model" in user_lower:
            # User explicitly wants to bypass CI and fit directly
            print(f"[DEBUG] User requested direct fit from dag_proposed; skipping CI")
            self.state.current_step = "dag_finalized"
            self.save_session_state()
            return await self._step_5_fit_model()
        else:
            # Unclear response - provide clear guidance
            response = "**I'm not sure what you'd like to do.**\n\n"
            response += "Here are your options:\n\n"
            
            actions = [
                {
                    "label": "Test the DAG against your data",
                    "command": "yes, test the model",
                    "description": "Run conditional independence tests to validate the DAG"
                },
                {
                    "label": "Modify the DAG structure",
                    "command": "open dag editor",
                    "description": "Use the visual DAG editor for manual edge changes"
                },
                {
                    "label": "Fit the model now (skip CI test)",
                    "command": "fit model",
                    "description": "Proceed directly to training with the current DAG"
                },
                {
                    "label": "Ask for a different DAG",
                    "command": "propose a different DAG",
                    "description": "I'll suggest an alternative causal structure"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response
    
    async def _step_3_test_dag(self) -> str:
        """Step 3: Test DAG consistency"""
        if not self.state.proposed_dag or not self.state.data_path:
            return "Error: DAG or data not available."
        
        response = "Testing DAG consistency with your data...\n\n"
        
        ci_results = self.tools.test_dag_consistency(
            self.state.proposed_dag,
            self.state.data_path,
            alpha=0.05
        )
        
        self.state.ci_test_results = ci_results
        
        consistent = ci_results.get('consistent', False)
        rejected_count = ci_results.get('rejected_count', 0)
        tests = ci_results.get('tests', [])
        total_tests = len(tests)
        passed_count = total_tests - rejected_count
        
        # Generate the DAG plot with CI test results (green/red lines + table)
        try:
            plot_path = self.tools._create_dag_plot_with_ci_results(
                self.session_id, self.state.proposed_dag, ci_results
            )
            if plot_path and Path(plot_path).exists():
                plot_filename = Path(plot_path).name
                response += f"![DAG CI Test Results](/api/plots/{plot_filename})\n\n"
                print(f"[INFO] DAG+CI results plot created: {plot_filename}")
            else:
                print(f"[WARNING] DAG+CI results plot not generated")
        except Exception as e:
            print(f"[WARNING] Failed to generate DAG+CI results plot: {e}")
        
        response += f"**Consistency Test Results:**\n\n"
        response += f"- **Consistent with data:** {'Yes' if consistent else 'No'}\n"
        response += f"- **Total CI tests:** {total_tests}\n"
        response += f"- **Passed:** {passed_count}\n"
        response += f"- **Rejected:** {rejected_count}\n"
        
        # List individual test results
        if tests:
            response += f"\n**Individual CI Test Details:**\n\n"
            for i, test in enumerate(tests, 1):
                ci_str = test.get('ci', 'Unknown')
                rejected = test.get('rejected', False)
                p_value = test.get('adj_p_value', test.get('p_value', None))
                test_name = str(
                    test.get('test_name')
                    or test.get('test')
                    or test.get('method')
                    or test.get('ci_test')
                    or test.get('test_type')
                    or 'CI'
                ).upper()
                
                status_icon = "REJECTED" if rejected else "Passed"
                p_str = f"p={p_value:.4f}" if p_value is not None else ""
                
                response += f"- `{test_name}` — {status_icon} (rejected={str(bool(rejected)).lower()}) — `{ci_str}` ({p_str})\n"
        
        if rejected_count > 0:
            response += "\n**Some conditional independence assumptions were rejected.**\n"
            response += "This means the DAG structure may not fully match the data.\n"
            response += "The **red dashed lines** in the plot above show which pairs failed the test.\n"
            
            actions = [
                {
                    "label": "Open DAG editor to fix",
                    "command": "open dag editor",
                    "description": "Use the visual editor to add/remove edges based on the test results"
                },
                {
                    "label": "Proceed anyway",
                    "command": "proceed with fitting",
                    "description": "Continue to model fitting despite inconsistencies (not recommended)"
                }
            ]
            response += self._format_action_suggestions(actions)
        else:
            response += "\n**The DAG is consistent with your data!**\n"
            response += "All conditional independence assumptions hold (shown as **green dashed lines** in the plot).\n"
            
            actions = [
                {
                    "label": "Fit the TRAM-DAG model",
                    "command": "fit model",
                    "description": "Train the model using this DAG structure (takes a few minutes)"
                },
                {
                    "label": "Modify the DAG anyway",
                    "command": "open dag editor",
                    "description": "Use the visual editor to make changes before fitting"
                }
            ]
            response += self._format_action_suggestions(actions)
        
        self.state.current_step = "dag_tested"
        # Update session immediately
        if self.session_id in sessions:
            sessions[self.session_id]["current_step"] = "dag_tested"
            sessions[self.session_id]["ci_test_results"] = self.state.ci_test_results
        print(f"[DEBUG] Set current_step to 'dag_tested' and saved to session")
        return response
    
    async def _step_4_propose_revisions(self) -> str:
        """
        Step 4: Propose DAG revisions based on CI test results (Agent Decision Only)
        
        NOTE: Only the revision suggestion uses the local LLM. The CI test results
        themselves were computed locally.
        """
        ci_results = self.state.ci_test_results
        rejected_count = ci_results.get('rejected_count', 0)
        
        # If DAG is consistent, skip revisions and go straight to fitting
        if rejected_count == 0:
            response = "**Great news!** Your DAG is consistent with the data.\n\n"
            response += "No revisions needed. You can proceed directly to model fitting.\n\n"
            
            # IMPORTANT: Transition to dag_finalized so user can proceed to fitting
            self.state.current_step = "dag_finalized"
            self.save_session_state()  # Save state immediately
            print(f"[DEBUG] DAG consistent - set step to 'dag_finalized' for session {self.session_id}")
            
            actions = [
                {
                    "label": "Fit the TRAM-DAG model",
                    "command": "fit model",
                    "description": "Train the model using this DAG structure (takes a few minutes)"
                },
                {
                    "label": "Modify the DAG anyway",
                    "command": "open dag editor",
                    "description": "Use the visual editor to make DAG changes before fitting"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response
        
        # Use LLM to suggest revisions (agent decision only - no data sent)
        sys_msg = """You are a causal inference expert. Analyze CI test failures and suggest DAG revisions."""
        
        rejected = ci_results.get('tests', [])
        rejected_cis = [t for t in rejected if t.get('rejected', False)]
        
        usr_msg = f"""The following conditional independence assumptions were rejected:
{json.dumps(rejected_cis[:5], indent=2)}

Current DAG edges:
{json.dumps(self.state.proposed_dag['edges'], indent=2)}

Suggest specific revisions to the DAG structure to address these failures. Format your response clearly without LaTeX notation. Use plain text for variables (e.g., x1, x2, x3 instead of \\(x1\\), \\(x2\\)), and keep the response concise and well-structured."""
        
        llm_response = llm_client.chat.completions.create(
            model=llm_model_interpretation,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}
            ],
            temperature=0.3
        )
        
        revision_suggestion = llm_response.choices[0].message.content
        
        # Clean up LLM response formatting
        revision_suggestion = self._clean_llm_response(revision_suggestion)
        
        response = "**Suggested DAG Revisions:**\n\n"
        response += revision_suggestion
        
        # Add action suggestions
        actions = [
            {
                "label": "Apply these revisions",
                "command": "apply revisions",
                "description": "I'll update the DAG based on the suggestions above"
            },
            {
                "label": "Manually modify the DAG",
                "command": "open dag editor",
                "description": "Use the visual editor to make DAG changes"
            },
            {
                "label": "Proceed to model fitting",
                "command": "fit model",
                "description": "Use the current DAG (with or without revisions) to fit the model"
            },
            {
                "label": "Test the DAG again",
                "command": "test DAG",
                "description": "Re-run consistency tests after making changes"
            }
        ]
        response += self._format_action_suggestions(actions)
        
        self.state.current_step = "dag_finalized"
        return response
    
    async def _apply_revisions_to_dag(self) -> str:
        """Apply LLM-suggested revisions to the DAG by asking the LLM to extract edge changes."""
        ci_results = self.state.ci_test_results
        if not ci_results:
            return "No CI test results available. Please test the DAG first."
        
        rejected_cis = [t for t in ci_results.get('tests', []) if t.get('rejected', False)]
        if not rejected_cis:
            response = "**No revisions needed** — your DAG passed all CI tests!\n\n"
            actions = [
                {
                    "label": "Fit the TRAM-DAG model",
                    "command": "fit model",
                    "description": "Train the model using this DAG structure"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response
        
        # Use LLM to suggest specific edge changes
        vars = self.state.proposed_dag.get("variables", [])
        current_edges = self.state.proposed_dag.get("edges", [])
        
        sys_msg = """You are a causal inference expert. Based on CI test failures, suggest specific edge additions.
Return ONLY a JSON object with:
{"add_edges": [["parent", "child"], ...], "remove_edges": [["parent", "child"], ...]}
Do not include any other text."""
        
        usr_msg = f"""Variables: {vars}
Current edges: {json.dumps(current_edges)}
Rejected CI tests: {json.dumps(rejected_cis[:5])}

Suggest specific edge additions/removals to fix the DAG. Return JSON only."""
        
        try:
            llm_response = llm_client.chat.completions.create(
                model=llm_model_interpretation,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg}
                ],
                temperature=0.1
            )
            
            response_text = llm_response.choices[0].message.content
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                changes = json.loads(json_match.group(0))
            else:
                changes = json.loads(response_text)
            
            add_edges = changes.get("add_edges", [])
            remove_edges = changes.get("remove_edges", [])
            
            # Apply changes
            n = len(vars)
            adj_matrix = np.zeros((n, n), dtype=int)
            var_to_idx = {v: i for i, v in enumerate(vars)}
            
            # Start with current edges
            for p, c in current_edges:
                if p in var_to_idx and c in var_to_idx:
                    adj_matrix[var_to_idx[p], var_to_idx[c]] = 1
            
            # Remove edges
            for edge in remove_edges:
                if len(edge) == 2 and edge[0] in var_to_idx and edge[1] in var_to_idx:
                    adj_matrix[var_to_idx[edge[0]], var_to_idx[edge[1]]] = 0
            
            # Add edges
            for edge in add_edges:
                if len(edge) == 2 and edge[0] in var_to_idx and edge[1] in var_to_idx and edge[0] != edge[1]:
                    adj_matrix[var_to_idx[edge[0]], var_to_idx[edge[1]]] = 1
            
            # Build updated edges list
            updated_edges = []
            for i, p in enumerate(vars):
                for j, c in enumerate(vars):
                    if adj_matrix[i, j] == 1:
                        updated_edges.append((p, c))
            
            # Update the DAG
            self.state.proposed_dag["adjacency_matrix"] = adj_matrix.tolist()
            self.state.proposed_dag["edges"] = updated_edges
            self._invalidate_runtime_after_dag_change()
            self.state.current_step = "dag_proposed"
            self.save_session_state()
            
            response = "**DAG revisions applied!**\n\n"
            if add_edges:
                response += "**Added edges:** " + ", ".join(f"{e[0]} → {e[1]}" for e in add_edges) + "\n"
            if remove_edges:
                response += "**Removed edges:** " + ", ".join(f"{e[0]} → {e[1]}" for e in remove_edges) + "\n"
            response += f"\n**Updated edges:**\n"
            for p, c in updated_edges:
                response += f"- {p} → {c}\n"
            
            actions = [
                {
                    "label": "Test the revised DAG",
                    "command": "yes, test the model",
                    "description": "Run CI tests to verify the revisions fixed the issues"
                },
                {
                    "label": "Modify further",
                    "command": "open dag editor",
                    "description": "Use the visual editor to make additional changes"
                },
                {
                    "label": "Fit the model",
                    "command": "fit model",
                    "description": "Skip re-testing and fit the model with this DAG"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response
            
        except Exception as e:
            print(f"[ERROR] Failed to apply revisions: {e}")
            response = "**Could not automatically apply revisions.**\n\n"
            response += "Please modify the DAG in the interactive editor:\n\n"
            actions = [
                {
                    "label": "Open DAG editor",
                    "command": "open dag editor",
                    "description": "Use the visual editor to make changes"
                },
            ]
            response += self._format_action_suggestions(actions)
            return response
    
    async def _step_5_fit_model(self) -> str:
        """Step 5: Fit TRAM-DAG model"""
        if not self.state.proposed_dag or not self.state.data_path:
            return "Error: DAG or data not available."

        # Use the unified tool execution path so fit behavior stays consistent
        # with MCP artifact-first logic and avoids legacy local plot rendering.
        fit_result = await self._execute_tool("fit_model", {})
        return await self._format_tool_results([("fit_model", fit_result)], "fit model")
    
    async def _step_6_answer_query(self, query: str) -> str:
        """Legacy step-6 entrypoint; delegate to unified agent/tool pipeline."""
        return await self._agent_select_and_execute_tool(query)

        experiment_dir = self.state.experiment_dir
        query_lower = query.lower()
        
        # Check if user is asking about workflow status
        if "what step" in query_lower or "where am i" in query_lower or "status" in query_lower:
            dag_vars = self.state.proposed_dag.get("variables", []) if self.state.proposed_dag else []
            if not dag_vars:
                dag_vars = self._example_vars()
            example_var = dag_vars[0] if dag_vars else "x1"
            example_outcome = dag_vars[-1] if len(dag_vars) > 1 else example_var
            
            response = "**Current Workflow Status:**\n\n"
            response += "- Data uploaded\n"
            response += "- DAG proposed\n"
            response += "- DAG tested\n"
            response += "- **Model fitted** (you are here)\n\n"
            response += "You can now ask causal questions, compute treatment effects, generate plots, or download reports.\n\n"
            
            actions = [
                {
                    "label": "Compute treatment effect",
                    "command": "compute the ate",
                    "description": "I will ask which treatment (X) and outcome (Y) variables to use"
                },
                {
                    "label": "Generate plots",
                    "command": "sample 1000 data points and show me the plots",
                    "description": "Visualize distributions and loss history"
                },
                {
                    "label": "Download report",
                    "command": "download report",
                    "description": "Get a comprehensive PDF report"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response
        
        # Check if user says "fit model" when already fitted
        if "fit" in query_lower and "model" in query_lower:
            dag_vars = self.state.proposed_dag.get("variables", []) if self.state.proposed_dag else []
            if not dag_vars:
                dag_vars = self._example_vars()
            example_var = dag_vars[0] if dag_vars else "x1"
            example_outcome = dag_vars[-1] if len(dag_vars) > 1 else example_var
            
            response = "**The model is already fitted!**\n\n"
            response += "You can now ask causal questions and perform analyses.\n\n"
            
            actions = [
                {
                    "label": "Compute treatment effect",
                    "command": "compute the ate",
                    "description": "I will ask which treatment (X) and outcome (Y) variables to use"
                },
                {
                    "label": "Generate plots",
                    "command": "sample 1000 data points and show me the plots",
                    "description": "Visualize distributions and loss history"
                },
                {
                    "label": "Download report",
                    "command": "download report",
                    "description": "Get a comprehensive PDF report"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response
        
        # Check for report generation requests
        if "download report" in query_lower or "generate report" in query_lower or "get report" in query_lower:
            try:
                reports_dir = REPORTS_DIR
                reports_dir.mkdir(exist_ok=True)
                report_path = reports_dir / f"{self.session_id}_report.pdf"
                CausalTools.generate_report(self.session_id, str(report_path), report_type="full")
                
                response = f"**PDF Report Generated!**\n\n"
                response += f"**Report saved to:** `{report_path}`\n\n"
                response += "**The report includes:**\n"
                response += "- DAG structure visualization\n"
                response += "- Training loss history\n"
                response += "- Sampling distributions\n"
                response += "- Intervention results (if available)\n"
                response += "- Complete analysis summary\n\n"
                response += "**Tip:** Click the 'Download Report' button in the header to download it directly!"
                
                self._set_session_report_path(report_path)
                
                # Add action suggestions
                actions = [
                    {
                        "label": "Ask more causal questions",
                        "command": "compute the ate",
                        "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                    },
                    {
                        "label": "Generate plots",
                        "command": "sample 1000 data points and show me the plots",
                        "description": "Create visualizations of distributions and loss history"
                    }
                ]
                response += self._format_action_suggestions(actions)
                
                return response
            except Exception as e:
                return f"**Error generating report:** {str(e)}\n\nPlease try again or check the server logs."
        
        # Check for plot requests (including "loss history", "show distributions", "sample N data points")
        if ("plot" in query_lower or "graph" in query_lower or "visualize" in query_lower or
                "loss" in query_lower or "distribution" in query_lower or
                ("sample" in query_lower and ("data point" in query_lower or "show" in query_lower))):
            return await self._handle_plot_request(query, experiment_dir)
        
        # Parse intervention queries (e.g., "what if x1 = 3", "effect of x1 on x3", "do(x1=1)")
        intervention_pattern = r'(?:what if|if|change|set|do\()\s*([a-zA-Z0-9_]+)\s*[=:]\s*([0-9.]+)'
        intervention_match = re.search(intervention_pattern, query_lower)
        
        # Parse ATE queries (e.g., "effect of x1 on x3", "what is the effect of X on Y")
        ate_pattern = r'(?:effect|impact|influence)\s+(?:of|on)\s+([a-zA-Z0-9_]+)\s+(?:on|to)\s+([a-zA-Z0-9_]+)'
        ate_match = re.search(ate_pattern, query_lower)
        
        if intervention_match or "intervention" in query_lower or "treatment" in query_lower:
            # Intervention query
            if intervention_match:
                var = intervention_match.group(1)
                value = float(intervention_match.group(2))
                
                # Sample with intervention
                try:
                    samples = self.tools.sample_from_model(
                        experiment_dir=experiment_dir,
                        n_samples=10000,
                        do_interventions={var: value}
                    )
                    
                    response = f"**Intervention Results: do({var} = {value})**\n\n"
                    response += "**Sampled Values:**\n"
                    
                    # Prepare data for explanation
                    samples_summary = {}
                    for node, data in samples.items():
                        values = data["values"]
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        response += f"- **{node}**: mean = {mean_val:.4f}, std = {std_val:.4f}\n"
                        samples_summary[node] = {"mean": mean_val, "std": std_val}
                    
                    # Generate medical-friendly explanation
                    try:
                        explanation_data = {
                            "variable": var,
                            "intervention_value": value,
                            "samples": samples_summary
                        }
                        explanation = await self._generate_medical_explanation("intervention", explanation_data)
                        if not explanation:
                            explanation = "This intervention simulation shows how the model output shifts when one variable is fixed to a chosen value."
                        response += f"\n**What This Means (Simple Explanation):**\n\n{explanation}\n\n"
                    except Exception as e:
                        print(f"[WARNING] Failed to generate explanation for intervention: {e}")
                    
                    # Add action suggestions
                    vars = self.state.proposed_dag.get("variables", [])
                    other_vars = [v for v in vars if v != var]
                    actions = [
                        {
                            "label": "Compute ATE",
                            "command": "compute the ate",
                            "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                        },
                        {
                            "label": "Try different intervention value",
                            "command": f"What if {var} = {value + 1}?",
                            "description": "See the effect with a different value"
                        },
                        {
                            "label": "Generate plots",
                            "command": "show me plots",
                            "description": "Visualize intervention effects"
                        }
                    ]
                    response += self._format_action_suggestions(actions)
                    response += self._calculation_followup_note()
                    
                    return response
                except Exception as e:
                    response = f"**Error computing intervention:** {str(e)}\n\n"
                    response += "**This might be because:**\n"
                    response += f"- Variable {var} doesn't exist in your DAG\n"
                    response += "- The model needs to be refitted\n"
                    response += "- There's an issue with the experiment directory\n\n"
                    response += "**You can try:**\n\n"
                    
                    dag_vars = self.state.proposed_dag.get("variables", [])
                    if dag_vars:
                        response += f"**Available variables:** {', '.join(dag_vars)}\n\n"
                        actions = [
                            {
                                "label": "Generate plots",
                                "command": "show me plots",
                                "description": "Check if the model is working correctly"
                            }
                        ]
                        response += self._format_action_suggestions(actions)
                    return response
            elif ate_match:
                # ATE query
                X = ate_match.group(1)
                Y = ate_match.group(2)
                
                # Try to extract treatment/control values from query
                x_treated = 1.0
                x_control = 0.0
                
                # Look for specific values in query
                treated_match = re.search(rf'{X}\s*[=:]\s*([0-9.]+)', query_lower)
                control_match = re.search(rf'control|baseline|without', query_lower)
                
                if treated_match:
                    x_treated = float(treated_match.group(1))
                
                try:
                    ate_result = self.tools.compute_ate(
                        experiment_dir=experiment_dir,
                        X=X,
                        Y=Y,
                        x_treated=x_treated,
                        x_control=x_control,
                        n_samples=10000
                    )
                    
                    # Add query info to result
                    ate_result['X'] = X
                    ate_result['Y'] = Y
                    ate_result['x_treated'] = x_treated
                    ate_result['x_control'] = x_control
                    
                    # Save to session for report generation
                    self._set_session_ate_result(ate_result)
                    
                    response = f"**Average Treatment Effect (ATE)**\n\n"
                    response += f"**Treatment**: do({X} = {x_treated})\n"
                    response += f"**Control**: do({X} = {x_control})\n"
                    response += f"**Outcome**: {Y}\n\n"
                    response += f"**Results:**\n"
                    response += f"- ATE = {ate_result['ate']:.4f}\n"
                    response += f"- E[{Y} | do({X}={x_treated})] = {ate_result['y_treated_mean']:.4f} (std: {ate_result['y_treated_std']:.4f})\n"
                    response += f"- E[{Y} | do({X}={x_control})] = {ate_result['y_control_mean']:.4f} (std: {ate_result['y_control_std']:.4f})\n\n"

                    # Auto-generate intervention (ATE) plot
                    try:
                        intervention_img_path = CausalTools._create_intervention_plot(experiment_dir, ate_result)
                        if intervention_img_path and Path(intervention_img_path).exists():
                            plot_filename = Path(intervention_img_path).name
                            response += "**Intervention (ATE) Plot:**\n"
                            response += f"![ATE](/api/plots/{plot_filename})\n\n"
                        else:
                            response += "Intervention plot could not be generated.\n\n"
                    except Exception as e:
                        print(f"[WARNING] Failed to generate intervention plot: {e}")
                        response += "Intervention plot could not be generated.\n\n"

                    # Auto-generate additional plots after ATE
                    try:
                        # Loss history
                        loss_img_path = CausalTools._create_loss_plot(experiment_dir, session_id=self.session_id)
                        if loss_img_path and Path(loss_img_path).exists():
                            plot_filename = Path(loss_img_path).name
                            response += "**Training Loss History:**\n"
                            response += f"![Loss History](/api/plots/{plot_filename})\n\n"
                    except Exception as e:
                        print(f"[WARNING] Failed to generate loss plot after ATE: {e}")

                    try:
                        # Sampling distributions
                        dist_img_path = CausalTools._create_distribution_plot(experiment_dir, self.session_id, n_samples=5000)
                        if dist_img_path and Path(dist_img_path).exists():
                            plot_filename = Path(dist_img_path).name
                            response += "**Sampling Distributions (5000 samples):**\n"
                            response += f"![Distributions](/api/plots/{plot_filename})\n\n"
                    except Exception as e:
                        print(f"[WARNING] Failed to generate distribution plot after ATE: {e}")

                    try:
                        # DAG visualization
                        if self.state.proposed_dag:
                            dag_img_path = CausalTools._create_dag_plot(self.session_id, self.state.proposed_dag)
                            if dag_img_path and Path(dag_img_path).exists():
                                plot_filename = Path(dag_img_path).name
                                response += "**DAG Structure:**\n"
                                response += f"![DAG](/api/plots/{plot_filename})\n\n"
                    except Exception as e:
                        print(f"[WARNING] Failed to generate DAG plot after ATE: {e}")
                    
                    # Generate medical-friendly explanation
                    try:
                        explanation = await self._generate_medical_explanation("ate", ate_result)
                        if not explanation:
                            explanation = self._default_ate_interpretation(ate_result)
                        response += f"**What This Means (Simple Explanation):**\n\n{explanation}\n\n"
                    except Exception as e:
                        print(f"[WARNING] Failed to generate explanation for ATE: {e}")
                        response += f"**What This Means (Simple Explanation):**\n\n{self._default_ate_interpretation(ate_result)}\n\n"
                    
                    # Auto-generate intervention report
                    try:
                        reports_dir = REPORTS_DIR
                        reports_dir.mkdir(exist_ok=True)
                        report_path = reports_dir / f"{self.session_id}_intervention_{X}_{Y}.pdf"
                        CausalTools.generate_report(self.session_id, str(report_path), report_type="intervention")
                        response += f"**Intervention report generated!** Download it via the 'Download Report' button.\n"
                        self._set_session_report_path(report_path)
                    except Exception as e:
                        print(f"[WARNING] Failed to generate intervention report: {e}")
                    
                    # Add action suggestions after ATE computation
                    vars = self.state.proposed_dag.get("variables", [])
                    other_vars = [v for v in vars if v != X and v != Y]
                    actions = [
                        {
                            "label": "Download intervention report",
                            "command": "download report",
                            "description": "Get PDF report with ATE results and intervention plots"
                        },
                        {
                            "label": "Try different intervention values",
                            "command": f"What if {X} = 2?",
                            "description": "See the effect with different treatment values"
                        },
                        {
                            "label": "Compute ATE for other variables",
                            "command": "compute the ate",
                            "description": "Choose a new treatment (X) and outcome (Y) pair"
                        },
                        {
                            "label": "Generate plots",
                            "command": "show me plots",
                            "description": "Visualize distributions, loss history, and intervention effects"
                        }
                    ]
                    response += self._format_action_suggestions(actions)
                    response += self._calculation_followup_note()
                    
                    return response
                except Exception as e:
                    response = f"**Error computing ATE:** {str(e)}\n\n"
                    response += "**This might be because:**\n"
                    response += f"- Variables {X} or {Y} don't exist in your DAG\n"
                    response += "- The model needs to be refitted\n"
                    response += "- There's an issue with the experiment directory\n\n"
                    dag_vars = self.state.proposed_dag.get("variables", [])
                    if dag_vars:
                        response += f"**Available variables:** {', '.join(dag_vars)}\n\n"
                        response += "**You can try:**\n\n"
                        example_var = dag_vars[0]
                        example_outcome = dag_vars[-1] if len(dag_vars) > 1 else dag_vars[0]
                        actions = [
                            {
                                "label": "Compute ATE with correct variables",
                                "command": "compute the ate",
                                "description": f"Use variables from your DAG: {', '.join(dag_vars)}"
                            },
                            {
                                "label": "Generate plots",
                                "command": "show me plots",
                                "description": "Check if the model is working correctly"
                            }
                        ]
                        response += self._format_action_suggestions(actions)
                    else:
                        response += "**No variables found in DAG. Please check your model setup.**\n"
                    return response
            else:
                # Generic intervention query - provide clear guidance
                vars = self.state.proposed_dag.get("variables", [])
                if not vars:
                    vars = self._example_vars()
                example_var = vars[0] if vars else "x1"
                example_outcome = vars[-1] if len(vars) > 1 else example_var
                
                response = "**I can compute intervention effects!**\n\n"
                response += "**What I need:**\n"
                response += "- Treatment variable (the cause)\n"
                response += "- Outcome variable (the effect)\n"
                response += "- Treatment and control values (optional, defaults to 1 and 0)\n\n"
                
                actions = [
                    {
                        "label": "Compute ATE",
                        "command": "compute the ate",
                        "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                    },
                    {
                        "label": "Generate plots",
                        "command": "sample 1000 data points and show me the plots",
                        "description": "Visualize distributions and intervention effects"
                    }
                ]
                response += self._format_action_suggestions(actions)
                return response
        
        elif "counterfactual" in query_lower:
            vars = self.state.proposed_dag.get("variables", [])
            if not vars:
                vars = self._example_vars()
            response = "**Counterfactual Analysis**\n\n"
            response += "Counterfactual queries answer: 'What would have happened if...?'\n\n"
            response += "**What I need:**\n"
            response += "- A specific observation (values for all variables)\n"
            response += "- An intervention to apply\n\n"
            response += f"**Example:** 'What would {vars[-1] if vars else 'x3'} be if {vars[0] if vars else 'x1'}=2, given that {vars[0] if vars else 'x1'}=1, {vars[1] if len(vars) > 1 else 'x2'}=0.5, {vars[-1] if vars else 'x3'}=1.2?'\n\n"
            response += "**This feature is coming soon!**\n\n"
            response += "For now, you can use intervention analysis (ATE) to answer causal questions.\n\n"
            
            actions = [
                {
                    "label": "Compute ATE instead",
                    "command": "compute the ate",
                    "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                },
                {
                    "label": "Perform intervention",
                    "command": f"What if {vars[0] if vars else 'x1'} = 2?",
                    "description": "See what happens when you intervene on a variable"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response
        
        elif "association" in query_lower or "correlation" in query_lower or "show associations" in query_lower:
            # Association query - compute from data
            if self.state.data_df is not None:
                df = self.state.data_df
                corr = df.corr()
                response = "**Association (Correlation) Matrix:**\n\n"
                response += "```\n" + corr.to_string() + "\n```\n\n"
                response += "**Quick Interpretation:**\n"
                response += "- Values close to 1 or -1 indicate strong associations\n"
                response += "- Values close to 0 indicate weak associations\n"
                response += "- Remember: association ≠ causation!\n\n"
                
                # Generate medical-friendly explanation
                try:
                    explanation_data = {
                        "correlation_matrix": corr.to_string()
                    }
                    explanation = await self._generate_medical_explanation("association", explanation_data)
                    if explanation:
                        response += f"**What This Means (Simple Explanation):**\n\n{explanation}\n\n"
                except Exception as e:
                    print(f"[WARNING] Failed to generate explanation for associations: {e}")
                
                # Add action suggestions
                vars = self.state.proposed_dag.get("variables", [])
                actions = [
                    {
                        "label": "Compute causal effects (ATE)",
                        "command": "compute the ate",
                        "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                    },
                    {
                        "label": "Generate plots",
                        "command": "show me plots",
                        "description": "Visualize distributions and relationships"
                    },
                    {
                        "label": "Download full report",
                        "command": "download report",
                        "description": "Get comprehensive PDF with all results"
                    }
                ]
                response += self._format_action_suggestions(actions)
                return response
            else:
                response = "**Data not available**\n\n"
                response += "I need the data to compute associations.\n\n"
                
                actions = [
                    {
                        "label": "Upload data",
                        "command": "Click the 'Upload Data' button above",
                        "description": "Upload your CSV or Excel file"
                    }
                ]
                response += self._format_action_suggestions(actions)
                return response
        
        else:
            # Try to parse as natural language query using LLM (agent decision only)
            try:
                result = await self._parse_and_answer_query_with_llm(query, experiment_dir)
                
                # Ensure we always return something
                if not result or result.strip() == "":
                    result = self._get_fallback_response_for_query(query)
                
                # Add action suggestions if not already present
                if "action" not in result.lower() and "button" not in result.lower():
                    dag_vars = self.state.proposed_dag.get("variables", [])
                    if dag_vars:
                        example_var = dag_vars[0]
                        example_outcome = dag_vars[-1] if len(dag_vars) > 1 else dag_vars[0]
                        actions = [
                            {
                                "label": "Try a different question",
                                "command": "compute the ate",
                                "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                            },
                            {
                                "label": "Generate plots",
                                "command": "show me plots",
                                "description": "Visualize distributions and intervention effects"
                            },
                            {
                                "label": "Download report",
                                "command": "download report",
                                "description": "Get PDF with all results and visualizations"
                            }
                        ]
                        result += "\n\n" + self._format_action_suggestions(actions)
                
                return result
            except Exception as e:
                print(f"[ERROR] Error in _step_6_answer_query fallback: {e}")
                import traceback
                traceback.print_exc()
                return self._get_fallback_response_for_query(query)
    
    async def _handle_plot_request(self, query: str, experiment_dir: str) -> str:
        """Handle requests for plots/visualizations - Actually generate them!"""
        query_lower = query.lower()
        
        # Parse number of samples if specified
        n_samples = 10000  # default
        sample_match = re.search(r'(\d+)\s*(?:sample|data point)', query_lower)
        if sample_match:
            n_samples = int(sample_match.group(1))
        
        response = "**Generating plots...**\n\n"
        ate_data = None
        if self.session_id in sessions:
            ate_data = sessions[self.session_id].get("query_results", {}).get("ate")
        
        try:
            # 1. Generate sampling distribution plots
            if "sample" in query_lower or "distribution" in query_lower or "data point" in query_lower:
                response += f"**Sampling {n_samples} data points from the model...**\n\n"
                
                # Sample from model
                samples = self.tools.sample_from_model(
                    experiment_dir=experiment_dir,
                    n_samples=n_samples
                )
                
                # Create distribution plots
                dist_img_path = CausalTools._create_distribution_plot(experiment_dir, self.session_id, n_samples=n_samples)
                
                if dist_img_path and Path(dist_img_path).exists():
                    plot_filename = Path(dist_img_path).name
                    response += f"**Distribution plots generated!**\n\n"
                    response += f"![Distributions](/api/plots/{plot_filename})\n\n"
                    
                    # Add summary statistics
                    response += "**Sampling Summary:**\n"
                    for var, data in samples.items():
                        values = data['values']
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        min_val = np.min(values)
                        max_val = np.max(values)
                        response += f"- **{var}**: mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]\n"
                    response += "\n"
                else:
                    response += "Distribution plots could not be generated.\n\n"
            
            # 2. Generate loss history plot
            if "loss" in query_lower or "training" in query_lower or "all" in query_lower:
                loss_img_path = CausalTools._create_loss_plot(experiment_dir, session_id=self.session_id)
                if loss_img_path and Path(loss_img_path).exists():
                    plot_filename = Path(loss_img_path).name
                    response += f"**Loss history plot generated!**\n\n"
                    response += f"![Loss History](/api/plots/{plot_filename})\n\n"
            
            # 3. Generate DAG plot if available
            if self.state.proposed_dag and ("dag" in query_lower or "graph" in query_lower or "all" in query_lower):
                dag_img_path = CausalTools._create_dag_plot(self.session_id, self.state.proposed_dag)
                if dag_img_path and Path(dag_img_path).exists():
                    plot_filename = Path(dag_img_path).name
                    response += f"**DAG visualization generated!**\n\n"
                    response += f"![DAG](/api/plots/{plot_filename})\n\n"

            # 4. Generate ATE / intervention plot if available
            if "ate" in query_lower or "intervention" in query_lower:
                if ate_data:
                    intervention_img_path = CausalTools._create_intervention_plot(experiment_dir, ate_data)
                    if intervention_img_path and Path(intervention_img_path).exists():
                        plot_filename = Path(intervention_img_path).name
                        response += f"**Intervention (ATE) plot generated!**\n\n"
                        response += f"![ATE](/api/plots/{plot_filename})\n\n"
                    else:
                        response += "Intervention plot could not be generated.\n\n"
                else:
                    response += "No ATE results found yet. Ask an ATE question first (e.g., 'ATE of X on Y') and then request the plot.\n\n"
            
            # If no specific request, generate all available plots
            if not any(keyword in query_lower for keyword in ["sample", "distribution", "loss", "training", "dag", "graph", "ate", "intervention"]):
                response += "**Generating all available plots...**\n\n"
                
                # Loss plot
                loss_img_path = CausalTools._create_loss_plot(experiment_dir, session_id=self.session_id)
                if loss_img_path and Path(loss_img_path).exists():
                    plot_filename = Path(loss_img_path).name
                    response += f"**Training Loss History:**\n"
                    response += f"![Loss History](/api/plots/{plot_filename})\n\n"
                
                # Distribution plot
                samples = self.tools.sample_from_model(experiment_dir=experiment_dir, n_samples=n_samples)
                dist_img_path = CausalTools._create_distribution_plot(experiment_dir, self.session_id, n_samples=n_samples)
                if dist_img_path and Path(dist_img_path).exists():
                    plot_filename = Path(dist_img_path).name
                    response += f"**Sampling Distributions:**\n"
                    response += f"![Distributions](/api/plots/{plot_filename})\n\n"
                
                # DAG plot
                if self.state.proposed_dag:
                    dag_img_path = CausalTools._create_dag_plot(self.session_id, self.state.proposed_dag)
                    if dag_img_path and Path(dag_img_path).exists():
                        plot_filename = Path(dag_img_path).name
                        response += f"**DAG Structure:**\n"
                        response += f"![DAG](/api/plots/{plot_filename})\n\n"

                # Intervention plot (ATE)
                if ate_data:
                    intervention_img_path = CausalTools._create_intervention_plot(experiment_dir, ate_data)
                    if intervention_img_path and Path(intervention_img_path).exists():
                        plot_filename = Path(intervention_img_path).name
                        response += f"**Intervention (ATE) Plot:**\n"
                        response += f"![ATE](/api/plots/{plot_filename})\n\n"
                
                response += "**All plots generated and displayed above!**\n\n"
            
            response += f"**All plots saved in:** `{TEMP_PLOTS_DIR}`\n"
            
            # Add action suggestions after plot generation
            # Get variables from DAG for action suggestions
            dag_vars = self.state.proposed_dag.get("variables", []) if self.state.proposed_dag else []
            if not dag_vars:
                dag_vars = self._example_vars()
            actions = [
                {
                    "label": "Download full PDF report",
                    "command": "download report",
                    "description": "Get a comprehensive report with all plots, results, and analysis"
                },
                {
                    "label": "Ask more causal questions",
                    "command": "compute the ate",
                    "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                },
                {
                    "label": "Generate more plots",
                    "command": "show loss history plot",
                    "description": "Create additional visualizations (loss history, DAG, etc.)"
                }
            ]
            response += self._format_action_suggestions(actions)
            
        except Exception as e:
            response += f"**Error generating plots:** {str(e)}\n"
            response += f"Please check that the model is fitted and experiment directory exists: `{experiment_dir}`"
            print(f"[ERROR] Plot generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return response
    
    async def _parse_and_answer_query_with_llm(self, query: str, experiment_dir: str) -> str:
        """Use LLM to parse natural language query and extract parameters (Agent Decision Only)"""
        # Get available variables from DAG
        dag_vars = self.state.proposed_dag.get("variables", [])
        
        sys_msg = """You are a causal inference assistant. Parse user queries to extract:
- Query type: "association", "intervention", "counterfactual", or "unclear"
- Variables involved (treatment X, outcome Y)
- Intervention values if specified
- Confidence level: "high", "medium", or "low"
- If unclear, suggest what information is needed

Return a JSON object with keys: query_type, X, Y, x_treated, x_control, confidence, clarification_needed, suggestions"""
        
        usr_msg = f"""Available variables: {', '.join(dag_vars) if dag_vars else 'None (please check)'}

User query: {query}

Parse this query and return ONLY valid JSON with the structure:
{{
  "query_type": "intervention" | "association" | "counterfactual" | "unclear",
  "X": "variable_name" or null,
  "Y": "variable_name" or null,
  "x_treated": number or null,
  "x_control": number or null,
  "confidence": "high" | "medium" | "low",
  "clarification_needed": "string explaining what's unclear" or null,
  "suggestions": ["suggestion1", "suggestion2"] or []
}}"""
        
        try:
            llm_response = llm_client.chat.completions.create(
                model=llm_model_decision,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg}
                ],
                temperature=0.1
            )
            
            response_text = llm_response.choices[0].message.content
            # Try to extract JSON from response (might be wrapped in markdown)
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(response_text)
            
            confidence = parsed.get("confidence", "medium")
            clarification_needed = parsed.get("clarification_needed")
            suggestions = parsed.get("suggestions", [])
            
            # If confidence is low or clarification is needed, ask for clarification
            if confidence == "low" or clarification_needed or parsed.get("query_type") == "unclear":
                response = "**I need a bit more information to help you.**\n\n"
                if clarification_needed:
                    response += f"**What I'm unsure about:** {clarification_needed}\n\n"
                
                response += "**Here's what I can help you with:**\n\n"
                
                dag_vars = self.state.proposed_dag.get("variables", [])
                if not dag_vars:
                    dag_vars = self._example_vars()
                example_var = dag_vars[0] if dag_vars else "x1"
                example_outcome = dag_vars[-1] if len(dag_vars) > 1 else example_var
                
                actions = [
                    {
                        "label": "Compute Average Treatment Effect (ATE)",
                        "command": "compute the ate",
                        "description": "I will ask which treatment (X) and outcome (Y) variables to use"
                    },
                    {
                        "label": "Generate plots",
                        "command": "sample 1000 data points and show me the plots",
                        "description": "Create visualizations of distributions, loss history, and DAG"
                    },
                    {
                        "label": "Download full report",
                        "command": "generate report",
                        "description": "Get a comprehensive PDF report with all results"
                    }
                ]
                
                if suggestions:
                    response += "**Based on your query, you might want to:**\n"
                    for suggestion in suggestions:
                        response += f"- {suggestion}\n"
                    response += "\n"
                
                response += self._format_action_suggestions(actions)
                return response
            
            # Execute based on parsed query
            if parsed.get("query_type") == "intervention" and parsed.get("X") and parsed.get("Y"):
                X = parsed["X"]
                Y = parsed["Y"]
                x_treated = parsed.get("x_treated", 1.0)
                x_control = parsed.get("x_control", 0.0)
                
                if X not in dag_vars or Y not in dag_vars:
                    response = f"**I couldn't find those variables in your DAG.**\n\n"
                    response += f"You mentioned: **{X}** and **{Y}**\n"
                    response += f"Available variables: **{', '.join(dag_vars) if dag_vars else 'None'}**\n\n"
                    response += "**Please try one of these:**\n\n"
                    
                    if dag_vars:
                        example_var = dag_vars[0]
                        example_outcome = dag_vars[-1] if len(dag_vars) > 1 else dag_vars[0]
                        actions = [
                            {
                                "label": "Compute ATE with correct variables",
                                "command": "compute the ate",
                                "description": f"Use variables from your DAG: {', '.join(dag_vars)}"
                            }
                        ]
                        response += self._format_action_suggestions(actions)
                    return response
                
                try:
                    ate_result = self.tools.compute_ate(
                        experiment_dir=experiment_dir,
                        X=X,
                        Y=Y,
                        x_treated=x_treated,
                        x_control=x_control,
                        n_samples=10000
                    )
                    
                    # Add query info to result
                    ate_result['X'] = X
                    ate_result['Y'] = Y
                    ate_result['x_treated'] = x_treated
                    ate_result['x_control'] = x_control
                    
                    # Save to session for report generation
                    self._set_session_ate_result(ate_result)
                    
                    response = f"**Query**: {query}\n\n"
                    response += f"**Interpreted as**: Average Treatment Effect (ATE) of {X} on {Y}\n\n"
                    response += f"**Results:**\n"
                    response += f"- **ATE** = {ate_result['ate']:.4f}\n"
                    response += f"- E[{Y} | do({X}={x_treated})] = {ate_result['y_treated_mean']:.4f} (std: {ate_result['y_treated_std']:.4f})\n"
                    response += f"- E[{Y} | do({X}={x_control})] = {ate_result['y_control_mean']:.4f} (std: {ate_result['y_control_mean']:.4f})\n\n"
                    
                    # Generate medical-friendly explanation
                    try:
                        explanation = await self._generate_medical_explanation("ate", ate_result)
                        if not explanation:
                            explanation = self._default_ate_interpretation(ate_result)
                        response += f"**What This Means (Simple Explanation):**\n\n{explanation}\n\n"
                    except Exception as e:
                        print(f"[WARNING] Failed to generate explanation for ATE: {e}")
                        response += f"**What This Means (Simple Explanation):**\n\n{self._default_ate_interpretation(ate_result)}\n\n"
                    
                    # Auto-generate intervention report
                    try:
                        reports_dir = REPORTS_DIR
                        reports_dir.mkdir(exist_ok=True)
                        report_path = reports_dir / f"{self.session_id}_intervention_{X}_{Y}.pdf"
                        CausalTools.generate_report(self.session_id, str(report_path), report_type="intervention")
                        response += f"**Intervention report generated!** Download it via the 'Download Report' button.\n"
                        self._set_session_report_path(report_path)
                    except Exception as e:
                        print(f"[WARNING] Failed to generate intervention report: {e}")
                    
                    # Add action suggestions
                    other_vars = [v for v in dag_vars if v != X and v != Y]
                    actions = [
                        {
                            "label": "Download intervention report",
                            "command": "download report",
                            "description": "Get PDF report with ATE results and intervention plots"
                        },
                        {
                            "label": "Try different intervention values",
                            "command": f"What if {X} = {x_treated + 1}?",
                            "description": "See the effect with different treatment values"
                        },
                        {
                            "label": "Compute ATE for other variables",
                            "command": "compute the ate",
                            "description": "Choose a new treatment (X) and outcome (Y) pair"
                        }
                    ]
                    response += self._format_action_suggestions(actions)
                    response += self._calculation_followup_note()
                    
                    return response
                except Exception as e:
                    response = f"**Error computing ATE:** {str(e)}\n\n"
                    response += "**This might be because:**\n"
                    response += "- The variables don't exist in the fitted model\n"
                    response += "- The model needs to be refitted\n"
                    response += "- There's an issue with the experiment directory\n\n"
                    response += "**You can try:**\n\n"
                    
                    actions = [
                        {
                            "label": "Generate plots",
                            "command": "show me plots",
                            "description": "Check if the model is working correctly"
                        }
                    ]
                    response += self._format_action_suggestions(actions)
                    return response
            else:
                # Query type is not intervention or missing variables
                query_type = parsed.get('query_type', 'unknown')
                response = f"**I understood your query as: {query_type}**\n\n"
                
                if not parsed.get("X") or not parsed.get("Y"):
                    response += "**I need more information:**\n"
                    if not parsed.get("X"):
                        response += "- Which variable is the treatment (cause)?\n"
                    if not parsed.get("Y"):
                        response += "- Which variable is the outcome (effect)?\n"
                    response += "\n"
                
                response += "**Here are some specific things you can ask:**\n\n"
                
                dag_vars = self.state.proposed_dag.get("variables", [])
                if not dag_vars:
                    dag_vars = self._example_vars()
                example_var = dag_vars[0] if dag_vars else "x1"
                example_outcome = dag_vars[-1] if len(dag_vars) > 1 else example_var
                
                actions = [
                    {
                        "label": "Compute Average Treatment Effect",
                        "command": "compute the ate",
                        "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                    },
                    {
                        "label": "Generate plots",
                        "command": "sample 1000 data points and show me the plots",
                        "description": "Create visualizations"
                    }
                ]
                response += self._format_action_suggestions(actions)
                return response
        except json.JSONDecodeError as e:
            # LLM didn't return valid JSON - ask for clarification
            response = "**I had trouble parsing your request.**\n\n"
            response += "**Could you rephrase your question?** Here are some examples:\n\n"
            
            dag_vars = self.state.proposed_dag.get("variables", [])
            if not dag_vars:
                dag_vars = self._example_vars()
            example_var = dag_vars[0] if dag_vars else "x1"
            example_outcome = dag_vars[-1] if len(dag_vars) > 1 else example_var
            
            actions = [
                {
                    "label": "Compute Average Treatment Effect",
                    "command": "compute the ate",
                    "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                },
                {
                    "label": "Generate plots",
                    "command": "sample 1000 data points and show me the plots",
                    "description": "Create visualizations"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response
        except Exception as e:
            # Any other error - provide helpful fallback
            print(f"[ERROR] LLM parsing failed: {e}")
            import traceback
            traceback.print_exc()
            
            response = "**I encountered an issue processing your request.**\n\n"
            response += "**Don't worry! Here are some things I can definitely help you with:**\n\n"
            
            dag_vars = self.state.proposed_dag.get("variables", [])
            if not dag_vars:
                dag_vars = self._example_vars()
            example_var = dag_vars[0] if dag_vars else "x1"
            example_outcome = dag_vars[-1] if len(dag_vars) > 1 else example_var
            
            actions = [
                {
                    "label": "Compute Average Treatment Effect",
                    "command": "compute the ate",
                    "description": "Choose treatment (X) and outcome (Y), then compute ATE"
                },
                {
                    "label": "Generate plots",
                    "command": "sample 1000 data points and show me the plots",
                    "description": "Create visualizations"
                },
                {
                    "label": "Download full report",
                    "command": "generate report",
                    "description": "Get a comprehensive PDF report"
                }
            ]
            response += self._format_action_suggestions(actions)
            return response


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def get_ui():
    """Serve the chat UI"""
    html_path = APP_DIR / "chatbot_ui.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("""
    <html>
        <head><title>Causal AI Agent</title></head>
        <body>
            <h1>Causal AI Agent Chatbot</h1>
            <p>UI file not found. Please create chatbot_ui.html</p>
        </body>
    </html>
    """)


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat messages"""
    session_id = request.session_id
    user_message = request.message
    
    print(f"[DEBUG] Chat endpoint called")
    print(f"[DEBUG] Received session_id: {session_id}")
    print(f"[DEBUG] Received message: {user_message[:100] if user_message else 'None'}...")
    
    if not session_id or session_id == "null" or session_id == "undefined":
        session_id = str(uuid.uuid4())
        print(f"[WARNING] No valid session_id provided, created new one: {session_id}")
        print(f"[WARNING] This means the frontend didn't pass the session_id from the upload!")
    else:
        print(f"[DEBUG] Using provided session_id: {session_id}")
    
    # Check if session exists and preserve existing data
    if session_id not in sessions:
        sessions[session_id] = {
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "current_step": "initial"
        }
        print(f"[DEBUG] Created NEW session: {session_id}")
    else:
        print(f"[DEBUG] Using EXISTING session: {session_id}")
        existing_session = sessions[session_id]
        print(f"[DEBUG] Existing session keys: {list(existing_session.keys())}")
        print(f"[DEBUG] Existing session current_step: {existing_session.get('current_step')}")
        print(f"[DEBUG] Existing session has data: {existing_session.get('data_df') is not None}")
        # IMPORTANT: Don't reset current_step if it already exists!
        # Only set to "initial" if it's truly a new session
    
    # Debug: log session state before processing
    session = sessions[session_id]
    print(f"[DEBUG] Processing message for session {session_id}")
    print(f"[DEBUG] All session keys: {list(session.keys())}")
    print(f"[DEBUG] Session has data_path: {session.get('data_path') is not None}")
    print(f"[DEBUG] Session has data_df: {session.get('data_df') is not None}")
    print(f"[DEBUG] Session has data_info: {session.get('data_info') is not None}")
    if session.get('data_info'):
        print(f"[DEBUG] Data info columns: {session['data_info'].get('columns')}")
    print(f"[DEBUG] All available session IDs: {list(sessions.keys())}")
    
    # If session has no data but we have other sessions with data, warn
    if not session.get('data_df') and len(sessions) > 1:
        sessions_with_data = [sid for sid, s in sessions.items() if s.get('data_df')]
        if sessions_with_data:
            print(f"[WARNING] Session {session_id} has no data, but other sessions do: {sessions_with_data}")
    
    # Create or get orchestrator (it will load session state automatically)
    print(f"[DEBUG] Creating orchestrator for session {session_id}")
    print(f"[DEBUG] Session current_step before orchestrator creation: {sessions[session_id].get('current_step')}")
    orchestrator = AgentOrchestrator(session_id)
    print(f"[DEBUG] Orchestrator created with current_step: {orchestrator.state.current_step}")
    
    # CRITICAL: Sync current_step from session to orchestrator state
    # The orchestrator should have restored it, but double-check
    session_step = sessions[session_id].get('current_step')
    if session_step and session_step != orchestrator.state.current_step:
        print(f"[WARNING] Step mismatch! Session has '{session_step}' but orchestrator has '{orchestrator.state.current_step}'")
        print(f"[WARNING] Syncing orchestrator state to match session")
        orchestrator.state.current_step = session_step
    
    # Refresh session data in orchestrator (in case it was updated)
    session = sessions[session_id]  # Get fresh session data
    if session.get("data_path") and not orchestrator.state.data_path:
        orchestrator.state.data_path = session["data_path"]
    if session.get("data_df") and orchestrator.state.data_df is None:
        try:
            orchestrator.state.data_df = pd.DataFrame(session["data_df"])
            print(f"[DEBUG] Refreshed data_df in orchestrator from session")
        except Exception as e:
            print(f"[ERROR] Failed to refresh data_df: {e}")
    if session.get("proposed_dag") and not orchestrator.state.proposed_dag:
        orchestrator.state.proposed_dag = session["proposed_dag"]
        print(f"[DEBUG] Refreshed proposed_dag in orchestrator from session")
    
    # Final check: ensure step is correct before processing
    latest_session_step = sessions[session_id].get('current_step', orchestrator.state.current_step)
    if latest_session_step != orchestrator.state.current_step:
        print(
            f"[WARNING] Final step sync: Setting orchestrator step from "
            f"'{orchestrator.state.current_step}' to '{latest_session_step}'"
        )
        orchestrator.state.current_step = latest_session_step
    
    print(f"[DEBUG] FINAL: About to process message with current_step='{orchestrator.state.current_step}'")
    print(f"[DEBUG] Session current_step at this moment: {sessions[session_id].get('current_step')}")
    
    # Process message
    try:
        response = await orchestrator.process_message(user_message)
    except Exception as e:
        import traceback
        error_msg = f"Error processing message: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print(traceback.format_exc())
        response = f"Error: {error_msg}\n\nPlease check the server logs for details."
    
    # Update session
    sessions[session_id]["messages"].append({
        "role": "user",
        "content": user_message,
        "timestamp": datetime.now().isoformat()
    })
    sessions[session_id]["messages"].append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })
    
    # Update state - CRITICAL: Save current_step to session
    sessions[session_id].update({
        "proposed_dag": orchestrator.state.proposed_dag,
        "ci_test_results": orchestrator.state.ci_test_results,
        "fitted_model": orchestrator.state.fitted_model,
        "experiment_dir": orchestrator.state.experiment_dir,
        "current_step": orchestrator.state.current_step,  # This must be saved!
        "user_question": user_message if orchestrator.state.current_step == "initial" else sessions[session_id].get("user_question", "")
    })
    
    # Debug: verify current_step was saved
    print(f"[DEBUG] Saved current_step to session: {orchestrator.state.current_step}")
    print(f"[DEBUG] Session current_step after save: {sessions[session_id].get('current_step')}")
    
    return {
        "session_id": session_id,
        "response": response,
        "current_step": orchestrator.state.current_step
    }


@app.post("/api/upload_data")
async def upload_data(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None)
):
    """Handle data upload"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Create uploads directory
    uploads_dir = UPLOADS_DIR
    uploads_dir.mkdir(exist_ok=True)
    
    # Save file
    file_path = uploads_dir / f"{session_id}_{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Load data to verify
    try:
        # Try CSV first
        try:
            df = pd.read_csv(file_path)
        except Exception as csv_error:
            # Try Excel
            try:
                import openpyxl  # For Excel support
                df = pd.read_excel(file_path)
            except ImportError:
                return {
                    "status": "error",
                    "error": "Excel files require openpyxl. Install with: pip install openpyxl"
                }
            except Exception as excel_error:
                return {
                    "status": "error",
                    "error": f"Could not read file. CSV error: {csv_error}. Excel error: {excel_error}"
                }
        
        # Store in session
        if session_id not in sessions:
            sessions[session_id] = {
                "session_id": session_id,
                "messages": [],
                "created_at": datetime.now().isoformat(),
                "current_step": "initial"
            }
        
        sessions[session_id]["data_path"] = str(file_path)
        sessions[session_id]["data_df"] = df.to_dict('records')  # Store as dict for JSON
        sessions[session_id]["data_info"] = {
            "shape": [int(df.shape[0]), int(df.shape[1])],  # Convert numpy int64 to Python int
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        data_pairplot_url = CausalTools._create_data_pairplot(session_id, df)
        sessions[session_id]["data_pairplot_url"] = data_pairplot_url
        
        # Debug: log upload success
        print(f"[DEBUG] Data uploaded for session {session_id}")
        print(f"[DEBUG] Data shape: {df.shape}")
        print(f"[DEBUG] Columns: {list(df.columns)}")
        print(f"[DEBUG] Session keys: {list(sessions[session_id].keys())}")
        print(f"[DEBUG] Has data_df: {sessions[session_id].get('data_df') is not None}")
        print(f"[DEBUG] Has data_info: {sessions[session_id].get('data_info') is not None}")
        
        return {
            "status": "uploaded",
            "session_id": session_id,
            "file_path": str(file_path),
            "data_info": sessions[session_id]["data_info"],
            "data_pairplot_url": data_pairplot_url,
        }
    except Exception as e:
        import traceback
        print(f"[ERROR] Upload failed: {e}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/api/upload_dag_csv")
async def upload_dag_csv(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Upload DAG edges from CSV and store as proposed DAG for an existing session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload data first.")

    session = sessions[session_id]
    if not session.get("data_info"):
        raise HTTPException(status_code=400, detail="No uploaded dataset found for this session.")

    uploads_dir = UPLOADS_DIR
    uploads_dir.mkdir(exist_ok=True)
    dag_file_path = uploads_dir / f"{session_id}_dag_{file.filename}"
    with open(dag_file_path, "wb") as f:
        f.write(await file.read())

    try:
        dag_df = pd.read_csv(dag_file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read DAG CSV: {e}")

    columns = {str(c).strip().lower(): c for c in dag_df.columns}
    source_col = None
    target_col = None

    source_candidates = ["source", "parent", "from", "src", "x"]
    target_candidates = ["target", "child", "to", "dst", "y"]
    for c in source_candidates:
        if c in columns:
            source_col = columns[c]
            break
    for c in target_candidates:
        if c in columns:
            target_col = columns[c]
            break

    parsed_edges: List[List[str]] = []
    if source_col and target_col:
        for _, row in dag_df[[source_col, target_col]].dropna().iterrows():
            parsed_edges.append([str(row[source_col]).strip(), str(row[target_col]).strip()])
    elif "edge" in columns:
        edge_col = columns["edge"]
        for value in dag_df[edge_col].dropna().astype(str):
            raw = value.strip()
            if "->" in raw:
                a, b = raw.split("->", 1)
            elif "," in raw:
                a, b = raw.split(",", 1)
            else:
                continue
            parsed_edges.append([a.strip(), b.strip()])
    else:
        raise HTTPException(
            status_code=400,
            detail="Could not parse DAG CSV. Use columns source,target (or parent,child), or an edge column with A->B."
        )

    if not parsed_edges:
        raise HTTPException(status_code=400, detail="No valid edges found in DAG CSV.")

    variables = session["data_info"].get("columns", [])
    var_set = set(variables)
    unknown_edges = [e for e in parsed_edges if e[0] not in var_set or e[1] not in var_set]
    if unknown_edges:
        raise HTTPException(
            status_code=400,
            detail=(
                "Some DAG edges reference variables not present in uploaded data. "
                f"Examples: {unknown_edges[:5]}"
            )
        )

    valid_edges = []
    seen = set()
    for parent, child in parsed_edges:
        if parent == child:
            continue
        key = (parent, child)
        if key in seen:
            continue
        seen.add(key)
        valid_edges.append((parent, child))

    if not valid_edges:
        raise HTTPException(status_code=400, detail="No valid non-self edges found after validation.")

    graph = nx.DiGraph()
    graph.add_nodes_from(variables)
    graph.add_edges_from(valid_edges)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        cycle_pairs = [[u, v] for u, v, _ in cycle]
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded DAG contains cycle(s): {cycle_pairs}. Please fix and re-upload."
        )

    adj = np.zeros((len(variables), len(variables)), dtype=int)
    idx = {v: i for i, v in enumerate(variables)}
    for parent, child in valid_edges:
        adj[idx[parent], idx[child]] = 1

    dag = {
        "adjacency_matrix": adj.tolist(),
        "variables": variables,
        "edges": valid_edges,
        "llm_explanation": "DAG imported from CSV file."
    }
    session["proposed_dag"] = dag
    invalidate_session_after_dag_update(session)
    session["current_step"] = "dag_proposed"

    plot_url = None
    try:
        plot_result = CausalTools._create_dag_plot_with_ci(session_id, dag)
        if plot_result:
            img_path, _ = plot_result
            plot_url = f"/api/plots/{Path(img_path).name}"
    except Exception as e:
        print(f"[WARNING] Could not generate DAG plot for uploaded DAG CSV: {e}")

    return {
        "status": "uploaded",
        "edge_count": len(valid_edges),
        "edges": [[a, b] for a, b in valid_edges],
        "plot_url": plot_url
    }


@app.get("/api/intervention_chooser_data")
async def intervention_chooser_data(session_id: str):
    """Return treatment-variable context plus metadata for intervention choice."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    pending_args = session.get("pending_tool_args") or {}
    treatment_var = pending_args.get("X")
    outcome_var = pending_args.get("Y")

    if not treatment_var:
        raise HTTPException(status_code=400, detail="No pending ATE treatment variable found")

    data_records = session.get("data_df")
    if not data_records:
        raise HTTPException(status_code=400, detail="No dataset available in session")

    df = pd.DataFrame(data_records)
    if treatment_var not in df.columns:
        raise HTTPException(status_code=400, detail=f"Treatment variable '{treatment_var}' not found in data")

    values = pd.to_numeric(df[treatment_var], errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        raise HTTPException(status_code=400, detail=f"Variable '{treatment_var}' has no numeric values for plotting")

    if len(values) > 3000:
        idx = np.linspace(0, len(values) - 1, 3000).astype(int)
        plot_values = values[idx]
    else:
        plot_values = values

    p10 = float(np.percentile(values, 10))
    p25 = float(np.percentile(values, 25))
    p50 = float(np.percentile(values, 50))
    p75 = float(np.percentile(values, 75))
    p90 = float(np.percentile(values, 90))

    return {
        "treatment_variable": treatment_var,
        "outcome_variable": outcome_var,
        "available_numeric_variables": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        "values": plot_values.tolist(),
        "summary": {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
        },
        "defaults": {
            "x_treated": 1.0,
            "x_control": 0.0,
        },
        "samples_vs_true_url": (session.get("mcp_plot_urls_by_kind", {}) or {}).get("samples_vs_true"),
    }


@app.get("/api/intervention_variable_distribution")
async def intervention_variable_distribution(session_id: str, variable: str):
    """Return histogram-ready values for any numeric variable in current session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    data_records = session.get("data_df")
    if not data_records:
        raise HTTPException(status_code=400, detail="No dataset available in session")

    df = pd.DataFrame(data_records)
    if variable not in df.columns:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found in data")

    values = pd.to_numeric(df[variable], errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        raise HTTPException(status_code=400, detail=f"Variable '{variable}' has no numeric values")

    if len(values) > 3000:
        idx = np.linspace(0, len(values) - 1, 3000).astype(int)
        plot_values = values[idx]
    else:
        plot_values = values

    return {
        "variable": variable,
        "values": plot_values.tolist(),
        "summary": {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "p10": float(np.percentile(values, 10)),
            "p25": float(np.percentile(values, 25)),
            "p50": float(np.percentile(values, 50)),
            "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)),
        },
    }


@app.get("/api/intervention_effect_preview")
async def intervention_effect_preview(
    session_id: str,
    x_treated: float,
    x_control: float,
    n_samples: int = 1000,
):
    """Preview expected variable shifts under selected intervention values."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    pending_args = session.get("pending_tool_args") or {}
    treatment_var = pending_args.get("X")
    outcome_var = pending_args.get("Y")
    experiment_dir = session.get("experiment_dir")
    if not treatment_var:
        raise HTTPException(status_code=400, detail="No pending treatment variable found")
    if not experiment_dir:
        raise HTTPException(status_code=400, detail="Model not fitted yet; fit model first for intervention preview")

    try:
        treated_samples = CausalTools.sample_from_model(
            experiment_dir=experiment_dir,
            n_samples=max(200, min(int(n_samples), 5000)),
            do_interventions={str(treatment_var): float(x_treated)},
            random_seed=42,
        )
        control_samples = CausalTools.sample_from_model(
            experiment_dir=experiment_dir,
            n_samples=max(200, min(int(n_samples), 5000)),
            do_interventions={str(treatment_var): float(x_control)},
            random_seed=43,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not compute intervention preview: {e}")

    common_vars = sorted(set(treated_samples.keys()).intersection(set(control_samples.keys())))
    rows: List[Dict[str, Any]] = []
    ordered_vars: List[str] = []
    data_records = session.get("data_df") or []
    if data_records:
        ordered_vars = [str(c) for c in pd.DataFrame(data_records).columns if c in common_vars]
    if not ordered_vars:
        ordered_vars = [str(v) for v in common_vars]

    numeric_for_plot: Dict[str, Dict[str, np.ndarray]] = {}
    for var in ordered_vars:
        treated_vals = pd.to_numeric(pd.Series(treated_samples.get(var, [])), errors="coerce").dropna()
        control_vals = pd.to_numeric(pd.Series(control_samples.get(var, [])), errors="coerce").dropna()
        if len(treated_vals) == 0 or len(control_vals) == 0:
            continue
        numeric_for_plot[str(var)] = {
            "treated": treated_vals.to_numpy(dtype=float),
            "control": control_vals.to_numpy(dtype=float),
        }
        treated_mean = float(treated_vals.mean())
        control_mean = float(control_vals.mean())
        delta = treated_mean - control_mean
        denom = abs(control_mean) if abs(control_mean) > 1e-9 else 1.0
        rows.append({
            "variable": str(var),
            "control_mean": control_mean,
            "treated_mean": treated_mean,
            "delta": delta,
            "relative_change": float(delta / denom),
            "is_treatment": str(var) == str(treatment_var),
            "is_outcome": bool(outcome_var) and str(var) == str(outcome_var),
        })

    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    plot_url = None
    try:
        plot_vars = [v for v in ordered_vars if v in numeric_for_plot]
        if plot_vars:
            n_vars = len(plot_vars)
            fig_w = max(4.5 * n_vars, 10.0)
            fig_h = 5.0
            fig, axes = plt.subplots(1, n_vars, figsize=(fig_w, fig_h))
            if n_vars == 1:
                axes = [axes]
            else:
                axes = list(axes)

            for idx, var in enumerate(plot_vars):
                ax = axes[idx]
                ctrl = numeric_for_plot[var]["control"]
                trt = numeric_for_plot[var]["treated"]
                all_vals = np.concatenate([ctrl, trt])
                if len(all_vals) == 0:
                    continue
                lo = float(np.min(all_vals))
                hi = float(np.max(all_vals))
                if not np.isfinite(lo) or not np.isfinite(hi):
                    continue
                if hi <= lo:
                    hi = lo + 1e-6
                bins = np.linspace(lo, hi, 80)
                ax.hist(ctrl, bins=bins, density=True, alpha=0.5, color="steelblue", label=f"do({treatment_var}={float(x_control):.4g})")
                ax.hist(trt, bins=bins, density=True, alpha=0.5, color="darkorange", label=f"do({treatment_var}={float(x_treated):.4g})")

                try:
                    from scipy.stats import gaussian_kde  # type: ignore
                    x_grid = np.linspace(lo, hi, 300)
                    if len(ctrl) > 1:
                        kde_c = gaussian_kde(ctrl)
                        ax.plot(x_grid, kde_c(x_grid), color="steelblue", lw=2, linestyle="--")
                    if len(trt) > 1:
                        kde_t = gaussian_kde(trt)
                        ax.plot(x_grid, kde_t(x_grid), color="darkorange", lw=2, linestyle="--")
                except Exception:
                    pass

                mean_c = float(np.mean(ctrl))
                mean_t = float(np.mean(trt))
                ax.axvline(x=mean_c, color="steelblue", linestyle=":", lw=1.5, alpha=0.8)
                ax.axvline(x=mean_t, color="darkorange", linestyle=":", lw=1.5, alpha=0.8)

                if str(var) == str(outcome_var):
                    ate_like = mean_t - mean_c
                    ax.set_title(f"{var}  (ATE = {ate_like:.4g})", fontsize=13, fontweight="bold")
                elif str(var) == str(treatment_var):
                    ax.set_title(f"{var}  (intervened)", fontsize=13, fontweight="bold")
                else:
                    shift = mean_t - mean_c
                    ax.set_title(f"{var}  (shift = {shift:.4g})", fontsize=12)

                ax.set_xlabel(str(var), fontsize=10)
                ax.set_ylabel("Density", fontsize=10)
                ax.legend(fontsize=8, loc="upper right")
                ax.tick_params(labelsize=9)

            fig.suptitle(
                f"Interventional Comparison: do({treatment_var}={float(x_control):.4g}) vs do({treatment_var}={float(x_treated):.4g})",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )
            fig.tight_layout()
            fname = f"{session_id}_intervention_preview_{uuid.uuid4().hex[:10]}.png"
            out_path = TEMP_PLOTS_DIR / fname
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            plot_url = f"/api/plots/{fname}"
    except Exception as plot_err:
        print(f"[WARNING] Could not generate intervention preview plot: {plot_err}")

    return {
        "treatment_variable": treatment_var,
        "outcome_variable": outcome_var,
        "x_treated": float(x_treated),
        "x_control": float(x_control),
        "plot_url": plot_url,
        "rows": rows[:25],
    }


@app.get("/api/debug_session")
async def debug_session(session_id: str):
    """Debug endpoint to check session state"""
    if session_id not in sessions:
        return {"error": "Session not found", "session_id": session_id}
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "has_data_path": session.get("data_path") is not None,
        "has_data_df": session.get("data_df") is not None,
        "has_data_info": session.get("data_info") is not None,
        "data_info": session.get("data_info"),
        "data_path": session.get("data_path"),
        "current_step": session.get("current_step"),
        "keys": list(session.keys())
    }


# ============================================================================
# DAG Editor API endpoints
# ============================================================================

@app.get("/api/dag_editor_data")
async def get_dag_editor_data(session_id: str):
    """Return current variables and edges for the interactive DAG editor"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Get variables from data_info or proposed_dag
    variables = []
    if session.get("proposed_dag"):
        variables = session["proposed_dag"].get("variables", [])
    elif session.get("data_info"):
        variables = session["data_info"].get("columns", [])
    
    # Get existing edges from proposed_dag (if any)
    edges: List[List[str]] = []
    if session.get("proposed_dag"):
        raw_edges = session["proposed_dag"].get("edges", [])
        edges = [[e[0], e[1]] for e in raw_edges if isinstance(e, (list, tuple)) and len(e) == 2]

    # Build richer node/edge payload for frontend editor.
    nodes = [{"id": v, "label": v} for v in variables]
    edge_objects = [{"source": e[0], "target": e[1]} for e in edges]

    ci_overlays = []
    ci_results = session.get("ci_test_results") or {}
    for t in ci_results.get("tests", []) or []:
        ci_text = str(t.get("ci", ""))
        pair_part = ci_text.split("|", 1)[0]
        if "_||_" in pair_part:
            left, right = pair_part.split("_||_", 1)
        elif "⟂" in pair_part:
            left, right = pair_part.split("⟂", 1)
        else:
            continue
        source = left.strip()
        target = right.strip()
        if source and target:
            is_rejected = bool(t.get("rejected"))
            ci_overlays.append({
                "source": source,
                "target": target,
                "status": "rejected" if is_rejected else "passed",
                "label": ci_text,
            })

    # If there are no CI test results yet, still show implied CI relations from
    # the current DAG so users see CI lines immediately after manual DAG setup.
    if not ci_overlays and session.get("proposed_dag"):
        try:
            implied_cis = CausalTools._compute_implied_cis(session["proposed_dag"])
            for ci in implied_cis:
                source = str(ci.get("x", "")).strip()
                target = str(ci.get("y", "")).strip()
                cond_set = ci.get("conditioning_set", []) or []
                if not source or not target:
                    continue
                label = f"{source} ⟂ {target}"
                if cond_set:
                    label += " | " + ", ".join(str(v) for v in cond_set)
                ci_overlays.append({
                    "source": source,
                    "target": target,
                    "status": "implied",
                    "label": label,
                })
        except Exception as exc:
            print(f"[WARNING] Could not compute implied CI overlays for DAG editor: {exc}")
    
    return {
        "variables": variables,
        "edges": edges,  # backwards-compatible: List[List[parent, child]]
        "nodes": nodes,
        "edge_objects": edge_objects,
        "ci_overlays": ci_overlays,
    }


@app.post("/api/save_dag")
async def save_dag(request: SaveDagRequest):
    """Save DAG from the interactive editor back to the session"""
    session_id = request.session_id
    edges = request.edges
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Get variables
    variables = []
    if session.get("proposed_dag"):
        variables = session["proposed_dag"].get("variables", [])
    elif session.get("data_info"):
        variables = session["data_info"].get("columns", [])
    
    if not variables:
        raise HTTPException(status_code=400, detail="No variables found in session")
    
    # Build adjacency matrix from edges + collect validation information
    n = len(variables)
    adj_matrix = np.zeros((n, n), dtype=int)
    var_to_idx = {v: i for i, v in enumerate(variables)}
    
    valid_edges = []
    invalid_edges = []
    self_edges = []
    seen_edges = set()
    for edge in edges:
        if len(edge) == 2:
            parent, child = edge[0], edge[1]
            if parent not in var_to_idx or child not in var_to_idx:
                invalid_edges.append([parent, child])
                continue
            if parent == child:
                self_edges.append([parent, child])
                continue
            key = (parent, child)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            adj_matrix[var_to_idx[parent], var_to_idx[child]] = 1
            valid_edges.append((parent, child))

    if invalid_edges or self_edges:
        return {
            "status": "validation_error",
            "message": "Some edges are invalid and were not accepted.",
            "details": {
                "invalid_edges": invalid_edges,
                "self_edges": self_edges,
            },
            "guidance": [
                "Use only variable names from the dataset.",
                "Self-edges like A -> A are not allowed in DAGs.",
            ],
        }

    # Validate acyclicity (DAG cannot contain directed cycles).
    graph = nx.DiGraph()
    graph.add_nodes_from(variables)
    graph.add_edges_from(valid_edges)
    if not nx.is_directed_acyclic_graph(graph):
        cycle = nx.find_cycle(graph, orientation="original")
        cycle_pairs = [[u, v] for u, v, _ in cycle]
        return {
            "status": "validation_error",
            "message": "This graph contains a directed cycle and cannot be saved as a DAG.",
            "details": {"cycle_edges": cycle_pairs},
            "guidance": [
                "Remove at least one edge from the cycle shown below.",
                "A valid DAG must be acyclic.",
            ],
        }
    
    # Update proposed_dag in session
    dag = {
        "adjacency_matrix": adj_matrix.tolist(),
        "variables": variables,
        "edges": valid_edges,
        "llm_explanation": "DAG built manually via interactive editor."
    }
    session["proposed_dag"] = dag
    # Invalidate old model artifacts; they belong to the previous DAG.
    invalidate_session_after_dag_update(session)
    session["current_step"] = "dag_proposed"
    
    print(f"[DEBUG] DAG saved from editor for session {session_id}: {len(valid_edges)} edges")
    
    # Generate DAG plot with implied CI tests
    plot_url = None
    implied_cis = []
    if valid_edges:
        try:
            result = CausalTools._create_dag_plot_with_ci(session_id, dag)
            if result is not None:
                img_path, implied_cis = result
                plot_filename = Path(img_path).name
                plot_url = f"/api/plots/{plot_filename}"
                print(f"[DEBUG] DAG+CI plot generated: {plot_url} with {len(implied_cis)} implied CIs")
        except Exception as e:
            print(f"[WARNING] Failed to generate DAG+CI plot: {e}")
            import traceback
            traceback.print_exc()
    
    # Format implied CIs for the response
    ci_list = []
    for ci in implied_cis:
        cond = ci["conditioning_set"]
        if cond:
            ci_list.append(f"{ci['x']} \u27C2 {ci['y']} | {', '.join(cond)}")
        else:
            ci_list.append(f"{ci['x']} \u27C2 {ci['y']}")
    
    return {
        "status": "saved",
        "variables": variables,
        "edges": valid_edges,
        "edge_count": len(valid_edges),
        "plot_url": plot_url,
        "implied_cis": ci_list,
        "feedback": f"Saved DAG with {len(valid_edges)} edges.",
        "next_actions": [
            {
                "label": "Test this DAG against your data",
                "command": "yes, test the model",
                "description": "Run conditional independence tests.",
            },
            {
                "label": "Open DAG editor again",
                "command": "open dag editor",
                "description": "Continue editing the DAG visually.",
            },
            {
                "label": "Fit model",
                "command": "fit model",
                "description": "Proceed to TRAM-DAG fitting.",
            },
        ],
    }


@app.get("/api/generate_report")
async def generate_report(session_id: str):
    """Generate and download PDF report"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    reports_dir = REPORTS_DIR
    reports_dir.mkdir(exist_ok=True)
    
    # Check if report already exists
    session = sessions.get(session_id, {})
    existing_report = session.get("report_path")
    
    if existing_report and Path(existing_report).exists():
        report_path = existing_report
    else:
        # Generate new report
        report_path = reports_dir / f"{session_id}_report.pdf"
        try:
            CausalTools.generate_report(session_id, str(report_path), report_type="full")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
    
    if not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"causal_analysis_report_{session_id}.pdf"
    )


@app.get("/api/plots/{filename}")
async def get_plot(filename: str):
    """Serve plot images for display in chat UI"""
    # Use absolute path constant to avoid path resolution issues
    plot_path = TEMP_PLOTS_DIR / filename
    
    # Security: only allow PNG files from temp_plots directory
    if not filename.endswith('.png'):
        raise HTTPException(status_code=400, detail="Only PNG files are allowed")
    
    # Ensure directory exists (already created at module load, but ensure it exists)
    TEMP_PLOTS_DIR.mkdir(exist_ok=True)
    
    # Debug: Log what we're looking for
    print(f"[DEBUG] API: Looking for plot: {plot_path}")
    print(f"[DEBUG] API: TEMP_PLOTS_DIR: {TEMP_PLOTS_DIR}")
    print(f"[DEBUG] API: TEMP_PLOTS_DIR exists: {TEMP_PLOTS_DIR.exists()}")
    
    # CRITICAL: Check if we have an open file handle for this file
    global _plot_file_handles
    if filename in _plot_file_handles:
        fh = _plot_file_handles[filename]
        try:
            # Check if file handle is still valid
            fh.seek(0)  # Reset to beginning
            file_data = fh.read()
            if file_data:
                print(f"[DEBUG] Serving plot from open file handle: {filename} ({len(file_data)} bytes)")
                return Response(
                    content=file_data,
                    media_type="image/png",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
                )
        except Exception as e:
            print(f"[WARNING] File handle for {filename} is invalid: {e}")
            # Remove invalid handle
            try:
                _plot_file_handles[filename].close()
            except:
                pass
            del _plot_file_handles[filename]
    
    # Force filesystem refresh before checking
    import time
    time.sleep(0.1)  # Small delay to allow filesystem to sync
    
    # Try to open the file directly if it exists
    if plot_path.exists():
        try:
            with open(plot_path, 'rb') as f:
                file_data = f.read()
            print(f"[DEBUG] Plot found on disk! Serving: {plot_path} ({len(file_data)} bytes)")
            return Response(
                content=file_data,
                media_type="image/png",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
            )
        except Exception as e:
            print(f"[ERROR] Failed to read plot file: {e}")
    
    # List available plots for debugging
    available_plots = list(TEMP_PLOTS_DIR.glob("*.png")) if TEMP_PLOTS_DIR.exists() else []
    print(f"[ERROR] Plot not found: {plot_path}")
    print(f"[DEBUG] Available plots in {TEMP_PLOTS_DIR}: {[p.name for p in available_plots]}")
    all_files = list(TEMP_PLOTS_DIR.iterdir()) if TEMP_PLOTS_DIR.exists() else []
    print(f"[DEBUG] All files in directory: {[f.name for f in all_files]}")
    print(f"[DEBUG] Directory exists: {TEMP_PLOTS_DIR.exists()}, is_dir: {TEMP_PLOTS_DIR.is_dir()}")
    # Try to read the directory directly
    try:
        dir_contents = os.listdir(str(TEMP_PLOTS_DIR))
        print(f"[DEBUG] os.listdir contents: {dir_contents}")
    except Exception as e:
        print(f"[DEBUG] os.listdir failed: {e}")
    
    raise HTTPException(status_code=404, detail=f"Plot not found: {filename}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    session_id = None
    
    try:
        while True:
            try:
                data = await websocket.receive_json()
            except json.JSONDecodeError as e:
                print(f"[ERROR] WebSocket JSON decode error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": "Invalid JSON format. Please check your message."
                })
                continue  # Continue the loop instead of breaking
            
            if data.get("type") == "init":
                # Use provided session_id or get from localStorage via client
                provided_session_id = data.get("session_id")
                if provided_session_id:
                    session_id = provided_session_id
                    print(f"[DEBUG] WebSocket init: Using provided session_id: {session_id}")
                else:
                    # Try to find a session with data (most recent upload)
                    sessions_with_data = [sid for sid, s in sessions.items() if s.get('data_df')]
                    if sessions_with_data:
                        # Use the most recently created session with data
                        session_id = sessions_with_data[-1]
                        print(f"[DEBUG] WebSocket init: No session_id provided, using session with data: {session_id}")
                    else:
                        session_id = str(uuid.uuid4())
                        print(f"[DEBUG] WebSocket init: Creating new session: {session_id}")
                
                await websocket.send_json({
                    "type": "init",
                    "session_id": session_id,
                    "message": "Connected. How can I help with your causal inference question?"
                })
            elif data.get("type") == "message":
                user_message = data.get("message")
                
                # CRITICAL: Check if session_id from message matches our current one
                message_session_id = data.get("session_id")
                if message_session_id:
                    if message_session_id != session_id:
                        print(f"[DEBUG] WebSocket: Message has different session_id. Updating from '{session_id}' to '{message_session_id}'")
                    session_id = message_session_id
                elif not session_id:
                    # No session_id at all - try to find one with data
                    sessions_with_data = [sid for sid, s in sessions.items() if s.get('data_df')]
                    if sessions_with_data:
                        session_id = sessions_with_data[-1]
                        print(f"[DEBUG] WebSocket: No session_id in message, using session with data: {session_id}")
                    else:
                        session_id = str(uuid.uuid4())
                        print(f"[DEBUG] WebSocket: No session_id, creating new: {session_id}")
                
                # Ensure session exists
                if session_id not in sessions:
                    sessions[session_id] = {
                        "session_id": session_id,
                        "messages": [],
                        "created_at": datetime.now().isoformat(),
                        "current_step": "initial"
                    }
                    print(f"[DEBUG] WebSocket: Created new session: {session_id}")
                else:
                    print(f"[DEBUG] WebSocket: Using existing session: {session_id}")
                
                # Create orchestrator (it will restore state from session)
                print(f"[DEBUG] WebSocket: Creating orchestrator for session {session_id}")
                print(f"[DEBUG] WebSocket: Session current_step before orchestrator: {sessions[session_id].get('current_step')}")
                print(f"[DEBUG] WebSocket: Session has data_df: {sessions[session_id].get('data_df') is not None}")
                orchestrator = AgentOrchestrator(session_id)
                print(f"[DEBUG] WebSocket: Orchestrator created with current_step: {orchestrator.state.current_step}")
                
                # Sync current_step from session (same as HTTP endpoint)
                session_step = sessions[session_id].get('current_step')
                if session_step and session_step != orchestrator.state.current_step:
                    print(f"[DEBUG] WebSocket: Syncing step from '{orchestrator.state.current_step}' to '{session_step}'")
                    orchestrator.state.current_step = session_step
                elif not session_step:
                    print(f"[WARNING] WebSocket: Session has no current_step! Defaulting to orchestrator's: {orchestrator.state.current_step}")
                
                # Refresh session data in orchestrator
                session = sessions[session_id]
                if session.get("data_path") and not orchestrator.state.data_path:
                    orchestrator.state.data_path = session["data_path"]
                if session.get("data_df") and orchestrator.state.data_df is None:
                    try:
                        orchestrator.state.data_df = pd.DataFrame(session["data_df"])
                    except Exception as e:
                        print(f"[ERROR] WebSocket: Failed to refresh data_df: {e}")
                if session.get("proposed_dag") and not orchestrator.state.proposed_dag:
                    orchestrator.state.proposed_dag = session["proposed_dag"]
                
                # Final sync before processing
                final_session_step = sessions[session_id].get('current_step')
                if final_session_step and final_session_step != orchestrator.state.current_step:
                    print(f"[CRITICAL] WebSocket: Final sync - setting step to '{final_session_step}'")
                    orchestrator.state.current_step = final_session_step
                
                print(f"[DEBUG] WebSocket: About to process message with current_step='{orchestrator.state.current_step}'")
                
                # Send status update based on current step
                status_messages = {
                    "initial": "Analyzing your question and proposing DAG structure...",
                    "dag_proposed": "Processing your response...",
                    "dag_tested": "Analyzing test results and suggesting revisions...",
                    "dag_finalized": "Preparing to fit the model...",
                    "model_fitted": "Processing your query...",
                    "sampled": "Reviewing sampling outputs...",
                    "ate_computed": "Processing your causal query..."
                }
                status_msg = status_messages.get(orchestrator.state.current_step, "Processing your request...")
                
                await websocket.send_json({
                    "type": "status",
                    "content": status_msg,
                    "current_step": orchestrator.state.current_step
                })
                
                # Process message with error handling
                try:
                    response = await orchestrator.process_message(user_message)
                    
                    # CRITICAL: Ensure we always have a response
                    if not response or response.strip() == "":
                        print(f"[WARNING] WebSocket: Empty response from process_message, using fallback")
                        response = orchestrator._get_fallback_response_for_query(user_message)
                    
                    # CRITICAL: Save state after processing message
                    orchestrator.save_session_state()
                    print(f"[DEBUG] WebSocket: Saved state after processing, current_step='{orchestrator.state.current_step}'")
                    
                    # Update session messages
                    sessions[session_id]["messages"].append({
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.now().isoformat()
                    })
                    sessions[session_id]["messages"].append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    await websocket.send_json({
                        "type": "message",
                        "role": "assistant",
                        "content": response,
                        "current_step": orchestrator.state.current_step
                    })
                except Exception as e:
                    # Catch errors during message processing - don't disconnect!
                    import traceback
                    error_msg = str(e)
                    print(f"[ERROR] Error processing message in session {session_id}: {error_msg}")
                    print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
                    
                    # Send helpful error message with suggestions but keep connection alive
                    try:
                        # Try to get a helpful fallback response
                        try:
                            fallback_response = orchestrator._get_fallback_response_for_query(user_message)
                        except:
                            # If fallback fails, use generic message
                            fallback_response = f"**I encountered an error processing your request.**\n\n"
                            fallback_response += f"**Error:** {error_msg}\n\n"
                            fallback_response += "**Please try:**\n"
                            fallback_response += "- Rephrasing your question\n"
                            fallback_response += "- Using one of the action buttons below\n"
                            fallback_response += "- Checking that your model is fitted (if asking causal questions)\n\n"
                            fallback_response += "**The connection is still active - you can try again!**"
                        
                        await websocket.send_json({
                            "type": "error",
                            "role": "assistant",
                            "content": fallback_response,
                            "current_step": orchestrator.state.current_step if hasattr(orchestrator, 'state') else "unknown"
                        })
                    except Exception as send_error:
                        print(f"[ERROR] Failed to send error message: {send_error}")
                        # If we can't send, connection is broken - break the loop
                        break
            else:
                # Unknown message type
                await websocket.send_json({
                    "type": "error",
                    "content": f"Unknown message type: {data.get('type')}"
                })
                
    except WebSocketDisconnect:
        print(f"[INFO] WebSocket disconnected for session {session_id}")
    except json.JSONDecodeError as e:
        print(f"[ERROR] WebSocket JSON decode error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": "Invalid JSON format. Please check your message."
            })
        except:
            pass  # Connection might be closed
    except Exception as e:
        # Catch ALL other exceptions to prevent disconnection
        import traceback
        error_msg = str(e)
        print(f"[ERROR] WebSocket error in session {session_id}: {error_msg}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        
        # Try to send error message to client
        try:
            await websocket.send_json({
                "type": "error",
                "role": "assistant",
                "content": f"An error occurred: {error_msg}\n\nPlease try again or refresh the page. The connection will remain active.",
                "error_details": error_msg if len(error_msg) < 200 else error_msg[:200] + "..."
            })
        except Exception as send_error:
            print(f"[ERROR] Failed to send error message to client: {send_error}")
            # Connection is likely closed, can't continue - exit the function
            return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
