from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict] = None


class SessionState(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    data_path: Optional[str] = None
    data_df: Optional[Any] = None
    proposed_dag: Optional[Dict] = None
    ci_test_results: Optional[Dict] = None
    fitted_model: Optional[Any] = None
    experiment_dir: Optional[str] = None
    query_type: Optional[str] = None  # "association", "intervention", "counterfactual"
    current_step: Optional[str] = None
    pending_tool: Optional[str] = None
    pending_tool_args: Optional[Dict[str, Any]] = None
    pending_missing_param: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class SaveDagRequest(BaseModel):
    session_id: str
    edges: List[List[str]]  # List of [parent, child] pairs
