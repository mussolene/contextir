from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from contextir import OllamaClient, OpenAICompatibleClient


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class ModelClientTests(unittest.TestCase):
    @patch("contextir.clients.urllib.request.urlopen")
    def test_ollama_client_is_callable_and_reports_usage(self, urlopen) -> None:
        urlopen.return_value = FakeResponse(
            {
                "message": {"content": " READY "},
                "prompt_eval_count": 17,
                "eval_count": 2,
                "prompt_eval_duration": 2_500_000,
                "eval_duration": 4_000_000,
            }
        )
        client = OllamaClient("qwen3:0.6b", max_output_tokens=12)

        response = client.complete("Reply with READY")

        self.assertEqual(response.text, "READY")
        self.assertEqual(response.prompt_tokens, 17)
        self.assertEqual(response.prompt_ms, 2.5)
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen3:0.6b")
        self.assertEqual(payload["options"]["num_predict"], 12)
        self.assertFalse(payload["think"])

    @patch("contextir.clients.urllib.request.urlopen")
    def test_openai_compatible_client_sends_bearer_token(self, urlopen) -> None:
        urlopen.return_value = FakeResponse(
            {
                "choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 1},
            }
        )
        client = OpenAICompatibleClient("local-model", api_key="secret-token")

        self.assertEqual(client("Complete the task"), "done")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertTrue(request.full_url.endswith("/v1/chat/completions"))

    def test_clients_require_model_name(self) -> None:
        with self.assertRaises(ValueError):
            OllamaClient(" ")
        with self.assertRaises(ValueError):
            OpenAICompatibleClient("")

    def test_clients_reject_unsafe_or_invalid_endpoint_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "http or https"):
            OllamaClient("model", base_url="file:///tmp/socket")
        with self.assertRaisesRegex(ValueError, "timeout"):
            OpenAICompatibleClient("model", timeout=0)
        with self.assertRaisesRegex(ValueError, "max_output_tokens"):
            OllamaClient("model", max_output_tokens=0)

    @patch("contextir.clients.urllib.request.urlopen")
    def test_client_reports_malformed_provider_response(self, urlopen) -> None:
        urlopen.return_value = FakeResponse({"unexpected": True})

        with self.assertRaisesRegex(RuntimeError, "invalid Ollama response"):
            OllamaClient("model").complete("prompt")


if __name__ == "__main__":
    unittest.main()
