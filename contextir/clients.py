from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelResponse:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    prompt_ms: float | None = None
    generation_ms: float | None = None
    latency_ms: float | None = None


class OllamaClient:
    """Dependency-free callable client for a local Ollama chat endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 180,
        context_length: int = 32768,
        max_output_tokens: int = 256,
        temperature: float = 0,
        seed: int = 42,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        validate_endpoint(base_url, timeout, max_output_tokens)
        if context_length < 1:
            raise ValueError("context_length must be positive")
        self.model = model
        self.url = base_url.rstrip("/") + "/api/chat"
        self.timeout = timeout
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.seed = seed

    def __call__(self, prompt: str) -> str:
        return self.complete(prompt).text

    def complete(self, prompt: str) -> ModelResponse:
        started = time.perf_counter()
        result = post_json(
            self.url,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "num_ctx": self.context_length,
                    "num_predict": self.max_output_tokens,
                },
            },
            timeout=self.timeout,
        )
        try:
            text = str(result["message"]["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"invalid Ollama response from {self.url}: {exc}") from exc
        return ModelResponse(
            text=text,
            prompt_tokens=as_int(result.get("prompt_eval_count")),
            output_tokens=as_int(result.get("eval_count")),
            prompt_ms=ns_to_ms(result.get("prompt_eval_duration")),
            generation_ms=ns_to_ms(result.get("eval_duration")),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


class OpenAICompatibleClient:
    """Dependency-free callable client for OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: str = "",
        timeout: float = 180,
        max_output_tokens: int = 256,
        temperature: float = 0,
        seed: int | None = 42,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        validate_endpoint(base_url, timeout, max_output_tokens)
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.seed = seed

    def __call__(self, prompt: str) -> str:
        return self.complete(prompt).text

    def complete(self, prompt: str) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        started = time.perf_counter()
        result = post_json(
            self.url,
            payload,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
        )
        usage = result.get("usage", {})
        try:
            text = str(result["choices"][0]["message"]["content"]).strip()
        except (IndexError, KeyError, TypeError) as exc:
            raise RuntimeError(f"invalid OpenAI-compatible response from {self.url}: {exc}") from exc
        return ModelResponse(
            text=text,
            prompt_tokens=as_int(usage.get("prompt_tokens")),
            output_tokens=as_int(usage.get("completion_tokens")),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"cannot reach model endpoint {url}: {exc.reason}") from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid model response from {url}: {exc}") from exc


def as_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def ns_to_ms(value: Any) -> float | None:
    return round(float(value) / 1_000_000, 3) if value is not None else None


def validate_endpoint(base_url: str, timeout: float, max_output_tokens: int) -> None:
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base_url must use http or https")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")


__all__ = ["ModelResponse", "OllamaClient", "OpenAICompatibleClient"]
