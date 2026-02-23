from __future__ import annotations

import json

import httpx


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _messages(self, system_prompt: str, user_prompt: str, history: list[dict] | None = None) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def chat(self, system_prompt: str, user_prompt: str, history: list[dict] | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": self._messages(system_prompt, user_prompt, history),
            "temperature": 0.5,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
            resp.raise_for_status()
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "").strip()

    def chat_stream(self, system_prompt: str, user_prompt: str, history: list[dict] | None = None):
        """Yield text deltas from OpenAI-compatible streaming response."""
        payload = {
            "model": self.model,
            "messages": self._messages(system_prompt, user_prompt, history),
            "temperature": 0.5,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_line = line[6:]
                    else:
                        data_line = line

                    if data_line.strip() == "[DONE]":
                        break

                    try:
                        obj = json.loads(data_line)
                    except json.JSONDecodeError:
                        continue

                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
