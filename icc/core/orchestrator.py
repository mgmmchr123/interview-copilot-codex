from __future__ import annotations

import logging
import os
import re
from threading import Thread
from time import perf_counter
from typing import Callable

from icc.llm.client import LlmClient


MAX_HISTORY = 4
LLM_MAX_TOKENS = 3000
logger = logging.getLogger(__name__)


class InterviewOrchestrator:
    def __init__(self, llm_client: LlmClient) -> None:
        self.llm_client = llm_client
        self.default_prompt = ""
        self.default_mode = "auto"
        self.conversation_history: list[dict] = []
        self.resume_context: str = ""
        self.debug_stream = os.getenv("ICC_DEBUG_STREAM", "").strip() == "1"
        self._request_started_at: float | None = None
        self._type_line_buffer = ""
        self._type_extracted = False

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

    def load_resume(self, path: str) -> None:
        from icc.core.resume_loader import load_resume

        self.resume_context = load_resume(path)

    def _trim_history(self) -> None:
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = self.conversation_history[-MAX_HISTORY:]

    def _system_prompt_template(self) -> str:
        return (
            "CRITICAL INSTRUCTION: The absolute first characters of your entire \n"
            "response MUST be [TYPE: followed immediately by the question type and ].\n"
            "Example: [TYPE:CONCEPTUAL_BASIC]\n"
            "Never skip this line. Never add any text before it. \n"
            "This is a hard requirement, not a suggestion.\n"
            "You are a senior backend engineer answering questions in a real technical interview.\n"
            "Candidate background — reference this actively in your answer, \n"
            "prefer specific projects and metrics over generic explanations:\n"
            "{resume}\n\n"
            "Speak in formal, professional English as if talking to a hiring panel.\n"
            "Use first person (I). No markdown. No bullet points.\n"
            "No filler phrases like \"great question\", \"absolutely\", \"certainly\", or \"of course\".\n"
            "Do not invent any experience not listed in the candidate background.\n"
            "Include at least one concrete metric or number where it naturally fits.\n"
            "Target length: 45 to 60 seconds of spoken delivery (roughly 100 to 130 words).\n"
            "Do not pad the answer - stop when the structure is complete.\n"
            "CODING\n"
            "Restate the problem and confirm key constraints. Explain your approach and the reasoning "
            "reasoning behind it. Write a clean Java implementation. State time and space\n"
            "complexity. Call out one edge case. Optionally mention one optimization or alternative.\n"
            "SYSTEM_DESIGN\n"
            "Clarify functional requirements and scale. Describe the high-level architecture.\n"
            "Explain the role of key components. Describe the data flow. Identify bottlenecks\n"
            "and how you would scale. State one deliberate trade-off and why you made it.\n"
            "BEHAVIORAL\n"
            "Briefly set the situation. State your task and responsibility. Describe your\n"
            "specific action and the decision you made. State the result with measurable impact\n"
            "if possible. Optionally reflect on what you learned.\n"
            "CONCEPTUAL_BASIC\n"
            "Give a one-line definition. Explain how it works simply. State when you would\n"
            "use it. Name one limitation or gotcha.\n"
            "CONCEPTUAL_DEEP\n"
            "Start with a simple explanation. Explain the internal mechanism. Describe a\n"
            "real-world failure case or production scenario. State the trade-offs.\n"
            "DEEP_DIVE\n"
            "Set the context briefly. Describe what you built or owned. Explain the key\n"
            "technical decision you made. Describe a challenge or failure you hit.\n"
            "State the trade-off.\n"
            "DEBUGGING\n"
            "Clarify the symptoms. Describe what metrics or logs you would check. Explain how you would "
            "would isolate the problem. State your hypothesis. Describe how you\n"
            "would validate it. Describe the fix.\n"
            "SCENARIO\n"
            "Clarify the situation and constraints. Describe your immediate mitigation.\n"
            "Explain how you would find the root cause. Describe the long-term fix.\n"
            "State what monitoring or prevention you would put in place.\n"
            "FOLLOW_UP\n"
            "Directly address the follow-up without re-introducing context. Be concise.\n"
            "If it requires a different structure, apply the matching type above.\n"
        )

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
            args=(prompt, on_chunk, on_complete, on_error, images_b64),
            daemon=True,
        )
        worker.start()

    def _run_request(
        self,
        prompt: str,
        on_chunk: Callable[[str], None],
        on_complete: Callable[[], None],
        on_error: Callable[[str], None],
        images_b64: list[str] | None = None,
    ) -> None:
        system_message, user_message = self.build_prompt(prompt=prompt)
        is_follow_up = "[This is a follow-up" in prompt
        history = self.conversation_history if is_follow_up else []
        self._type_line_buffer = ""
        self._type_extracted = False
        try:
            full_response: list[str] = []
            buffer = ""
            for chunk in self.llm_client.stream_answer(
                prompt=user_message,
                system=system_message,
                history=history,
                images_b64=images_b64,
                model="gpt-4.1-mini",
                max_tokens=LLM_MAX_TOKENS,
            ):
                self._debug_log("enqueue_chunk", f"chars={len(chunk)} text={chunk!r}")
                chunk_to_render = chunk
                if not self._type_extracted:
                    self._type_line_buffer += chunk
                    if "\n" in self._type_line_buffer:
                        type_line, remainder = self._type_line_buffer.split("\n", 1)
                        match = re.fullmatch(r"\[TYPE:([A-Z_]+)\]", type_line.strip())
                        if match:
                            logger.info("LLM self-classified as: %s", match.group(1))
                        else:
                            logger.warning("LLM did not return a type line")
                            remainder = self._type_line_buffer
                        self._type_extracted = True
                        self._type_line_buffer = ""
                        chunk_to_render = remainder
                    elif len(self._type_line_buffer) > 60:
                        logger.warning("LLM did not return a type line")
                        self._type_extracted = True
                        chunk_to_render = self._type_line_buffer
                        self._type_line_buffer = ""
                    else:
                        chunk_to_render = ""

                buffer += chunk_to_render
                if len(buffer) > 24:
                    on_chunk(buffer)
                    buffer = ""
                full_response.append(chunk)
            if buffer:
                on_chunk(buffer)
        except RuntimeError as exc:
            on_error(str(exc))
            return

        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append(
            {"role": "assistant", "content": "".join(full_response)}
        )
        if is_follow_up:
            self._trim_history()
        else:
            self.conversation_history = self.conversation_history[-2:]
        on_complete()

    def build_prompt(self, prompt: str) -> tuple[str, str]:
        question = prompt.strip() or self.default_prompt
        resume = self.resume_context or "No resume provided."
        system_message = self._system_prompt_template().format(resume=resume)
        user_message = f"Question: {question}"
        return system_message, user_message
