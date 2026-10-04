"""Système 2 chat clients. Mistral and OpenRouter both expose the OpenAI chat-completions API."""

import asyncio
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from lawhack.config import PROVIDERS, Settings


@dataclass
class ChatResult:
    text: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class ChatClient(Protocol):
    async def complete(self, messages: list[dict], json_mode: bool = False) -> ChatResult: ...


class OpenAICompatibleChat:
    def __init__(self, model: str, provider: str = "mistral", api_key: str | None = None, temperature: float = 0.0, timeout: float = 60):
        base_url, key_env = PROVIDERS[provider]
        self.url = f"{os.environ.get(f'{provider.upper()}_BASE_URL') or base_url}/chat/completions"
        self.api_key = api_key or os.environ.get(key_env)
        if not self.api_key:
            raise RuntimeError(f"{key_env} manquant : renseignez-le dans .env")
        self.model, self.temperature, self.timeout = model, temperature, timeout

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    async def complete(self, messages: list[dict], json_mode: bool = False) -> ChatResult:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = await asyncio.to_thread(self._post, payload)
        usage = data.get("usage") or {}
        return ChatResult(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


def answer_client(settings: Settings) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(settings.answer_model, settings.answer_provider, temperature=settings.temperature)


def baseline_client(settings: Settings) -> OpenAICompatibleChat:
    return OpenAICompatibleChat(settings.baseline_model, settings.baseline_provider, temperature=settings.temperature)
