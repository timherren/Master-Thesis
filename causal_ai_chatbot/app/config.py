from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Runtime directories used by the chatbot service.
RUNTIME_DIR = APP_DIR / "runtime"
UPLOADS_DIR = RUNTIME_DIR / "uploads"
REPORTS_DIR = RUNTIME_DIR / "reports"
TEMP_PLOTS_DIR = RUNTIME_DIR / "temp_plots"
MCP_ARTIFACTS_CACHE_DIR = RUNTIME_DIR / "mcp_artifacts_cache"
EXPERIMENTS_BASE_DIR = RUNTIME_DIR / "experiments"

# Optional in-folder R bridge (not required for MCP wrapper flow).
R_INTEGRATION_DIR = APP_DIR / "r_integration"


def ensure_runtime_dirs() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MCP_ARTIFACTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_BASE_DIR.mkdir(parents=True, exist_ok=True)
