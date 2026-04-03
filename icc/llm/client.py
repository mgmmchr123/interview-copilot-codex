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
        self._last_stream_model: str | None = None
        self._last_stream_chunk_count = 0
        self._last_stream_exception_type: str | None = None
        self._last_stream_exception_message: str | None = None

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
        stop: list[str] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[str]:
        yield from self._stream_openai(
            prompt,
            system,
            history,
            images_b64,
            model,
            max_tokens,
            stop,
            api_key,
            base_url,
            timeout,
        )

    def complete_text(
        self,
        prompt: str,
        system: str = "",
        history: list[dict] | None = None,
        images_b64: list[str] | None = None,
        model: str | None = None,
        max_tokens: int = 400,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> str:
        return self._complete_openai(
            prompt=prompt,
            system=system,
            history=history,
            images_b64=images_b64,
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

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
        stop: list[str] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[str]:
        import openai

        selected_model = model or self.config.openai_model
        self._last_stream_model = selected_model
        self._last_stream_chunk_count = 0
        self._last_stream_exception_type = None
        self._last_stream_exception_message = None
        self._stream_started_at = perf_counter()
        self._debug_log(
            "request_start",
            f"prompt_chars={len(prompt)} model={selected_model}",
        )

        client = openai.OpenAI(
            api_key=api_key or self.config.openai_api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

        if selected_model == "gpt-5.4-mini" and images_b64 and base_url is None:
            yield from self._stream_openai_responses(
                client=client,
                prompt=prompt,
                system=system,
                history=history,
                images_b64=images_b64,
                model=selected_model,
                max_tokens=max_tokens,
            )
            return

        messages = self._build_messages(prompt, system, history, images_b64)

        try:
            request_kwargs: dict[str, object] = {
                "model": selected_model,
                "messages": messages,
                "stream": True,
                "max_tokens": max_tokens,
                "stream_options": {"include_usage": True},
            }
            if stop is not None:
                request_kwargs["stop"] = stop
            stream = client.chat.completions.create(
                **request_kwargs,
            )
            final_usage = None
            for chunk in stream:
                self._last_stream_chunk_count += 1
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    final_usage = usage
                if not chunk.choices:
                    continue
                if self.debug_stream:
                    delta_obj = chunk.choices[0].delta
                    print(f"[chunk-debug] delta={delta_obj!r}")
                content = chunk.choices[0].delta.content
                if content is not None and content != "":
                    self._debug_log("yield_chunk", f"chars={len(content)} text={content!r}")
                    yield content
            if final_usage is not None:
                logger.info(
                    "OpenAI usage: completion_tokens=%s total_tokens=%s",
                    getattr(final_usage, "completion_tokens", None),
                    getattr(final_usage, "total_tokens", None),
                )
        except openai.OpenAIError as exc:
            self._last_stream_exception_type = type(exc).__name__
            self._last_stream_exception_message = str(exc)
            raise RuntimeError(f"OpenAI error: {exc}") from exc

    def _stream_openai_responses(
        self,
        client,
        prompt: str,
        system: str,
        history: list[dict] | None,
        images_b64: list[str],
        model: str,
        max_tokens: int,
    ) -> Iterator[str]:
        import openai

        response_input = self._build_responses_input(prompt, history, images_b64)
        self._last_stream_model = model
        self._last_stream_chunk_count = 0
        self._last_stream_exception_type = None
        self._last_stream_exception_message = None

        try:
            stream = client.responses.create(
                model=model,
                instructions=system or None,
                input=response_input,
                stream=True,
                max_output_tokens=max_tokens,
            )
            final_usage = None
            for event in stream:
                self._last_stream_chunk_count += 1
                if getattr(event, "type", "") == "response.output_text.delta":
                    delta = getattr(event, "delta", "")
                    if delta:
                        self._debug_log("yield_chunk", f"chars={len(delta)} text={delta!r}")
                        yield delta
                elif getattr(event, "type", "") == "response.completed":
                    final_usage = getattr(getattr(event, "response", None), "usage", None)
            if final_usage is not None:
                logger.info(
                    "OpenAI usage: completion_tokens=%s total_tokens=%s",
                    getattr(final_usage, "output_tokens", None),
                    getattr(final_usage, "total_tokens", None),
                )
        except openai.OpenAIError as exc:
            self._last_stream_exception_type = type(exc).__name__
            self._last_stream_exception_message = str(exc)
            raise RuntimeError(f"OpenAI error: {exc}") from exc

    def _complete_openai(
        self,
        prompt: str,
        system: str = "",
        history: list[dict] | None = None,
        images_b64: list[str] | None = None,
        model: str | None = None,
        max_tokens: int = 400,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> str:
        import openai

        selected_model = model or self.config.openai_model
        client = openai.OpenAI(
            api_key=api_key or self.config.openai_api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )
        messages = self._build_messages(prompt, system, history, images_b64)

        try:
            response = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
            )
        except openai.OpenAIError as exc:
            raise RuntimeError(f"OpenAI error: {exc}") from exc

        if not response.choices:
            return ""

        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        return str(content)

    def _build_messages(
        self,
        prompt: str,
        system: str = "",
        history: list[dict] | None = None,
        images_b64: list[str] | None = None,
    ) -> list[dict[str, object]]:
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
        return messages

    def _build_responses_input(
        self,
        prompt: str,
        history: list[dict] | None = None,
        images_b64: list[str] | None = None,
    ) -> list[dict[str, object]]:
        input_items: list[dict[str, object]] = []

        for item in history or []:
            role = str(item.get("role", "")).strip()
            content = item.get("content", "")
            if role not in {"user", "assistant", "system", "developer"}:
                continue
            if isinstance(content, str):
                input_items.append({"role": role, "content": content})

        user_content: list[dict[str, object]] = []
        for image_b64 in images_b64 or []:
            user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_b64}",
                    "detail": "high",
                }
            )
        user_content.append({"type": "input_text", "text": prompt})
        input_items.append({"role": "user", "content": user_content})
        return input_items
