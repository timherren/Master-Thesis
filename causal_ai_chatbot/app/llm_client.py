from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import requests


class LocalLLMClient:
    """
    Minimal chat-completions client for local Ollama endpoint.
    """

    def __init__(self, base_url: str, api_key: str = "ollama") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        response = requests.post(url, headers=headers, json=payload, timeout=timeout or 60)
        if response.status_code >= 400:
            raise RuntimeError(f"Local LLM request failed ({response.status_code}): {response.text}")

        data = response.json()
        return self._to_response_obj(data)

    @staticmethod
    def _to_response_obj(data: Dict[str, Any]) -> Any:
        choices_out = []
        for choice in data.get("choices", []):
            message = choice.get("message", {})
            tool_calls_out = []
            for tc in message.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, dict):
                    args = json.dumps(args)
                if args is None:
                    args = "{}"
                tool_calls_out.append(
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name=fn.get("name", ""),
                            arguments=args,
                        )
                    )
                )
            message_obj = SimpleNamespace(
                content=message.get("content"),
                tool_calls=tool_calls_out if tool_calls_out else None,
            )
            choices_out.append(SimpleNamespace(message=message_obj))
        return SimpleNamespace(choices=choices_out)

