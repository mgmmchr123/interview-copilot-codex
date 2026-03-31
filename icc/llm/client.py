from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from time import perf_counter

import requests

from config import AppConfig


logger = logging.getLogger(__name__)


class LlmClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.provider = config.llm_provider
        self.debug_stream = os.getenv("ICC_DEBUG_STREAM", "").strip() == "1"
        self._stream_started_at: float | None = None

    def _debug_log(self, stage: str, detail: str) -> None:
        if not self.debug_stream:
            return

        if self._stream_started_at is None:
            elapsed_ms = 0.0
        else:
            elapsed_ms = (perf_counter() - self._stream_started_at) * 1000

        print(f"[stream-debug][client][{elapsed_ms:8.1f} ms] {stage}: {detail}")

    def stream_answer(
        self,
        prompt: str,
        system: str = "",
        history: list[dict] | None = None,
        images_b64: list[str] | None = None,
        model: str | None = None,
        max_tokens: int = 400,
    ) -> Iterator[str]:
        if self.provider == "openai":
            yield from self._stream_openai(
                prompt,
                system,
                history,
                images_b64,
                model,
                max_tokens,
            )
            return
        yield from self._stream_ollama(f"{system}\n\n{prompt}" if system else prompt)

    def _stream_ollama(self, prompt: str) -> Iterator[str]:
        self._stream_started_at = perf_counter()
        self._debug_log(
            "request_start",
            f"prompt_chars={len(prompt)} model={self.config.ollama_model}",
        )
        try:
            response = requests.post(
                f"{self.config.ollama_base_url.rstrip('/')}/api/generate",
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                },
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                f"Make sure Ollama is running at {self.config.ollama_base_url}."
            ) from exc
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Ollama did not respond in time. Try again or use a smaller model."
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        response.encoding = "utf-8"

        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line:
                continue

            self._debug_log("raw_line", f"chars={len(line)}")

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Ollama returned invalid streaming JSON.") from exc

            error_text = str(payload.get("error", "")).strip()
            if error_text:
                raise RuntimeError(f"Ollama error: {error_text}")

            chunk = str(payload.get("response", ""))
            if chunk:
                self._debug_log("yield_chunk", f"chars={len(chunk)} text={chunk!r}")
                yield chunk

    def _stream_openai(
        self,
        prompt: str,
        system: str = "",
        history: list[dict] | None = None,
        images_b64: list[str] | None = None,
        model: str | None = None,
        max_tokens: int = 400,
    ) -> Iterator[str]:
        import openai

        selected_model = model or self.config.openai_model
        self._stream_started_at = perf_counter()
        self._debug_log(
            "request_start",
            f"prompt_chars={len(prompt)} model={selected_model}",
        )

        client = openai.OpenAI(api_key=self.config.openai_api_key)
        messages: list[dict[str, object]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history or [])

        if images_b64:
            user_content_items: list[dict[str, object]] = []
            for image_b64 in images_b64:
                user_content_items.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    }
                )
            user_content_items.append({"type": "text", "text": prompt})
            user_content: str | list[dict[str, object]] = user_content_items
        else:
            user_content = prompt

        messages.append({"role": "user", "content": user_content})

        try:
            stream = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    logger.info(
                        "OpenAI usage: completion_tokens=%s total_tokens=%s",
                        getattr(usage, "completion_tokens", None),
                        getattr(usage, "total_tokens", None),
                    )
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    self._debug_log("yield_chunk", f"chars={len(delta)} text={delta!r}")
                    yield delta
        except openai.OpenAIError as exc:
            raise RuntimeError(f"OpenAI error: {exc}") from exc
