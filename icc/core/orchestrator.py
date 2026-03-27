from __future__ import annotations

import os
from threading import Thread
from time import perf_counter
from typing import Callable

from icc.llm.client import LlmClient


class InterviewOrchestrator:
    def __init__(self, llm_client: LlmClient) -> None:
        self.llm_client = llm_client
        self.default_prompt = ""
        self.default_mode = "auto"
        self.conversation_history: list[dict] = []
        self.resume_context: str = ""
        self.prompt_templates = {
            "auto": (
                "You are an interview coach for software engineers.\n"
                "Answer the interview question below in spoken English.\n"
                "Rules: no markdown, no bullet points, no headers.\n"
                "Automatically detect the question type and adjust accordingly:\n"
                "- Behavioral question: use natural STAR flow, focus on what YOU did and the result.\n"
                "- System design question: clarify assumptions, propose design, mention one trade-off.\n"
                "- Coding question: explain the approach, mention complexity, call out one edge case.\n"
                "- General/background question: be direct and concise.\n"
                "Length: 3 to 5 sentences for simple questions, up to 150 words for complex ones.\n"
                "Tone: confident, conversational, natural spoken English.\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
        }
        self.debug_stream = os.getenv("ICC_DEBUG_STREAM", "").strip() == "1"
        self._request_started_at: float | None = None

    def _debug_log(self, stage: str, detail: str) -> None:
        if not self.debug_stream:
            return

        if self._request_started_at is None:
            elapsed_ms = 0.0
        else:
            elapsed_ms = (perf_counter() - self._request_started_at) * 1000

        print(f"[stream-debug][orchestrator][{elapsed_ms:8.1f} ms] {stage}: {detail}")

    def clear_history(self) -> None:
        self.conversation_history = []

    def request_answer(
        self,
        prompt: str,
        mode: str,
        on_chunk: Callable[[str], None],
        on_complete: Callable[[], None],
        on_error: Callable[[str], None],
        images_b64: list[str] | None = None,
    ) -> None:
        self._request_started_at = perf_counter()
        worker = Thread(
            target=self._run_request,
            args=(prompt, mode, on_chunk, on_complete, on_error, images_b64),
            daemon=True,
        )
        worker.start()

    def _run_request(
        self,
        prompt: str,
        mode: str,
        on_chunk: Callable[[str], None],
        on_complete: Callable[[], None],
        on_error: Callable[[str], None],
        images_b64: list[str] | None = None,
    ) -> None:
        final_prompt = self.build_prompt(prompt=prompt, mode=mode)
        if not self.conversation_history and self.resume_context:
            self.conversation_history.append(
                {
                    "role": "system",
                    "content": (
                        "You are an interview coach for a software engineer.\n"
                        f"Candidate background:\n{self.resume_context}"
                    ),
                }
            )
        try:
            full_response: list[str] = []
            for chunk in self.llm_client.stream_answer(
                prompt=final_prompt,
                history=self.conversation_history,
                images_b64=images_b64,
            ):
                self._debug_log("enqueue_chunk", f"chars={len(chunk)} text={chunk!r}")
                on_chunk(chunk)
                full_response.append(chunk)
        except RuntimeError as exc:
            on_error(str(exc))
            return

        self.conversation_history.append({"role": "user", "content": final_prompt})
        self.conversation_history.append(
            {"role": "assistant", "content": "".join(full_response)}
        )
        on_complete()

    def build_prompt(self, prompt: str, mode: str) -> str:
        raw_question = prompt.strip() or self.default_prompt
        selected_mode = mode if mode in self.prompt_templates else self.default_mode
        template = self.prompt_templates[selected_mode]
        resume_section = (
            f"Candidate background:\n{self.resume_context}\n\n"
            if self.resume_context
            else ""
        )
        return resume_section + template.format(question=raw_question)
