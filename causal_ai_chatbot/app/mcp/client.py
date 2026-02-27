from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

from .contracts import MCPToolResult
from ..config import MCP_ARTIFACTS_CACHE_DIR


class LocalMCPClient:
    """
    Local MCP adapter.

    The chatbot calls this client to execute tools from:
    - DAG_Validator_Agent (R wrapper)
    - tram_dag_application (Python wrapper)
    """

    def __init__(self, chatbot_dir: Path) -> None:
        self.chatbot_dir = chatbot_dir.resolve()
        self.enabled = os.getenv("USE_MCP_TOOLS", "1").lower() not in {"0", "false", "no"}
        self._artifact_cache_dir = MCP_ARTIFACTS_CACHE_DIR
        self._artifact_cache_dir.mkdir(parents=True, exist_ok=True)

        wrappers_dir = self.chatbot_dir / "mcp_wrappers"
        self._dag_wrapper = wrappers_dir / "dag_validator_wrapper.R"
        self._tram_wrapper = wrappers_dir / "tram_wrapper.py"

    def is_ready(self) -> bool:
        if not self.enabled:
            return False
        return self._dag_wrapper.exists() and self._tram_wrapper.exists()

    def call(self, tool_name: str, payload: Dict[str, Any]) -> MCPToolResult:
        """
        Execute a MCP tool and return normalized payload.
        """
        if not self.enabled:
            return MCPToolResult(success=False, error="MCP client disabled")

        validation_error = self._validate_payload(tool_name, payload)
        if validation_error:
            return MCPToolResult(success=False, error=validation_error)

        if tool_name in {"propose_dag", "test_dag"}:
            return self._run_wrapper(
                command=["Rscript", str(self._dag_wrapper)],
                tool_name=tool_name,
                payload=payload,
                timeout_sec=180,
            )

        if tool_name in {"fit_model", "sample", "compute_ate"}:
            return self._run_wrapper(
                command=["python", str(self._tram_wrapper)],
                tool_name=tool_name,
                payload=payload,
                timeout_sec=1800,
            )

        if tool_name == "health":
            return MCPToolResult(
                success=self.is_ready(),
                data={
                    "enabled": self.enabled,
                    "dag_wrapper": str(self._dag_wrapper),
                    "dag_wrapper_exists": self._dag_wrapper.exists(),
                    "tram_wrapper": str(self._tram_wrapper),
                    "tram_wrapper_exists": self._tram_wrapper.exists(),
                },
            )

        return MCPToolResult(success=False, error=f"Unknown MCP tool: {tool_name}")

    def _validate_payload(self, tool_name: str, payload: Dict[str, Any]) -> str | None:
        required_map = {
            "propose_dag": ["vars"],
            "test_dag": ["dag", "data_path"],
            "fit_model": ["dag", "data_path", "experiment_dir"],
            "sample": ["experiment_dir"],
            "compute_ate": ["experiment_dir", "X", "Y"],
        }
        required = required_map.get(tool_name, [])
        missing = [k for k in required if payload.get(k) in (None, "")]
        if missing:
            return f"MCP payload missing required fields for {tool_name}: {', '.join(missing)}"
        return None

    def _run_wrapper(
        self,
        command: list[str],
        tool_name: str,
        payload: Dict[str, Any],
        timeout_sec: int,
    ) -> MCPToolResult:
        with tempfile.TemporaryDirectory(prefix="mcp_call_") as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.json"
            output_path = tmp_path / "output.json"
            artifact_dir = tmp_path / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)

            body = {
                "tool": tool_name,
                "payload": payload,
                "artifact_dir": str(artifact_dir),
            }
            input_path.write_text(json.dumps(body), encoding="utf-8")

            full_cmd = command + ["--input", str(input_path), "--output", str(output_path)]
            try:
                proc = subprocess.run(
                    full_cmd,
                    cwd=str(self.chatbot_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_sec,
                    text=True,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return MCPToolResult(success=False, error=f"MCP wrapper timeout for {tool_name}")

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                return MCPToolResult(
                    success=False,
                    error=f"MCP wrapper failed ({tool_name}): {stderr or stdout or 'unknown error'}",
                )

            if not output_path.exists():
                return MCPToolResult(success=False, error=f"MCP wrapper produced no output for {tool_name}")

            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return MCPToolResult(success=False, error=f"Invalid MCP JSON output: {exc}")

            result = MCPToolResult.from_dict(payload)

            # Persist artifacts out of the temporary wrapper directory before it is removed.
            # Otherwise, downstream import code receives paths that no longer exist.
            normalized_artifacts = []
            artifact_path_map: Dict[str, str] = {}
            for p in result.artifacts:
                artifact_path = Path(p)
                if artifact_path.exists():
                    safe_name = f"{tool_name}_{uuid.uuid4().hex}_{artifact_path.name}"
                    persisted_path = self._artifact_cache_dir / safe_name
                    try:
                        shutil.copy2(artifact_path, persisted_path)
                        normalized_artifacts.append(str(persisted_path))
                        artifact_path_map[str(artifact_path)] = str(persisted_path)
                    except Exception:
                        # Keep robust behavior: if copy fails, skip artifact.
                        pass
            result.artifacts = normalized_artifacts

            # Keep artifact_manifest paths aligned with persisted artifact locations.
            if isinstance(result.data, dict) and isinstance(result.data.get("artifact_manifest"), list):
                normalized_manifest = []
                for entry in result.data.get("artifact_manifest", []):
                    if isinstance(entry, dict):
                        updated = dict(entry)
                        original_path = str(updated.get("path", ""))
                        if original_path in artifact_path_map:
                            updated["path"] = artifact_path_map[original_path]
                        normalized_manifest.append(updated)
                result.data["artifact_manifest"] = normalized_manifest
            return result

