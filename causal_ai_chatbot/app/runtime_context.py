from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .llm_client import LocalLLMClient
from .mcp import LocalMCPClient
from .config import (
    APP_DIR,
    TEMP_PLOTS_DIR,
    R_INTEGRATION_DIR,
    ensure_runtime_dirs,
)


# Load environment variables
load_dotenv()


# Import TRAM-DAG components from installed package.
try:
    from tramdag import TramDagConfig, TramDagModel, TramDagDataset
except ImportError as e:
    TramDagConfig = None
    TramDagModel = None
    TramDagDataset = None
    print(
        "Warning: tramdag package not available. "
        "MCP wrapper tools can still run, but legacy in-process tramdag paths are disabled. "
        f"Details: {e}"
    )


# Optional in-folder R bridge (not required for MCP wrapper flow)
application_dir = R_INTEGRATION_DIR
if application_dir.exists():
    sys.path.insert(0, str(application_dir))
    try:
        from r_python_bridge import RConsistencyChecker
    except (ImportError, Exception) as e:
        RConsistencyChecker = None
        print(f"Warning: R consistency checker not available: {e}")
        print("  The chatbot will work without R integration. CI tests will be skipped.")
else:
    RConsistencyChecker = None
    print(f"Warning: R integration directory not found at {application_dir}")
    print(f"  Calculated path: {application_dir}")


# Runtime dirs
ensure_runtime_dirs()
print(f"[DEBUG] Temp plots directory set to: {TEMP_PLOTS_DIR}")

# LLM client (local Ollama only)
llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
if llm_provider != "ollama":
    raise ValueError("Only local Ollama is supported. Set LLM_PROVIDER=ollama")

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
llm_client = LocalLLMClient(base_url=ollama_base_url, api_key=ollama_api_key)
llm_model_decision = (
    os.getenv("LLM_MODEL_DECISION")
    or os.getenv("LLM_MODEL")
    or "qwen2.5:3b-instruct"
)
llm_model_interpretation = (
    os.getenv("LLM_MODEL_INTERPRETATION")
    or os.getenv("LLM_MODEL")
    or os.getenv("OLLAMA_MODEL")
    or "qwen2.5:7b-instruct"
)

print(
    "[DEBUG] LLM provider: "
    f"{llm_provider}, decision_model: {llm_model_decision}, "
    f"interpretation_model: {llm_model_interpretation}"
)

# R checker + MCP client
r_checker = RConsistencyChecker() if RConsistencyChecker else None
mcp_client = LocalMCPClient(APP_DIR)

# Keep file handles open to avoid transient file visibility issues
plot_file_handles = {}
