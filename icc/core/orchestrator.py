from __future__ import annotations

import logging
import os
import string
from threading import Thread
from time import perf_counter
from typing import Callable

from icc.llm.client import LlmClient


MAX_HISTORY = 4
logger = logging.getLogger(__name__)


def classify_question(text: str, has_images: bool = False) -> str:
    normalized = text.strip().lower()
    words = normalized.split()
    if has_images and len(words) < 10:
        return "coding"

    sanitized = normalized.translate(str.maketrans("", "", string.punctuation))

    technical_terms = [
        "java",
        "spring",
        "kafka",
        "rabbitmq",
        "postgresql",
        "oracle",
        "elasticsearch",
        "mongodb",
        "docker",
        "kubernetes",
        "aws",
        "ec2",
        "rds",
        "s3",
        "rest",
        "grpc",
        "oauth",
        "jwt",
        "multithreading",
        "concurrency",
        "deadlock",
        "race condition",
        "heap",
        "stack",
        "garbage collection",
        "jvm",
        "big o",
        "binary search",
        "hashmap",
        "linked list",
        "tree",
        "graph",
        "array",
        "latency",
        "throughput",
        "sharding",
        "replication",
        "load balancer",
    ]

    def starts_with(prefixes: list[str]) -> bool:
        return any(sanitized.startswith(prefix) for prefix in prefixes)

    def contains_any(phrases: list[str]) -> bool:
        return any(phrase in sanitized for phrase in phrases)

    has_technical_term = contains_any(technical_terms)

    if starts_with([
        "what about",
        "how about",
        "and if",
        "but what if",
        "follow up",
        "what if",
        "can you elaborate",
        "tell me more",
    ]):
        return "follow_up"

    debugging_terms = [
        "debug",
        "slow",
        "latency",
        "error",
        "exception",
        "memory leak",
        "cpu",
        "timeout",
        "not working",
        "crash",
        "outage",
        "stack trace",
        "logs",
        "metrics",
        "issue",
        "problem",
    ]
    diagnostic_phrases = ["how do you", "what would you", "why is"]
    if contains_any(debugging_terms) and contains_any(diagnostic_phrases):
        return "debugging"

    if contains_any([
        "how would you",
        "what would you do",
        "imagine",
        "suppose",
        "if you had to",
        "walk me through how you would",
    ]):
        return "scenario"

    if contains_any([
        "how did you",
        "tell me about your",
        "walk me through your",
        "in your experience",
        "in your current",
        "at your company",
        "your implementation",
        "your system",
        "your approach",
    ]):
        return "deep_dive"

    if contains_any([
        "design",
        "architecture",
        "scale",
        "distributed",
        "system",
        "service",
        "pipeline",
        "infrastructure",
    ]):
        return "system_design"

    behavioral_terms = [
        "tell me about a time",
        "give me an example",
        "describe a situation",
        "conflict",
        "challenge",
        "failure",
        "disagreement",
        "worked with",
        "difficult",
    ]
    if contains_any(behavioral_terms):
        return "behavioral"
    if sanitized.startswith("tell me about") and not has_technical_term:
        return "behavioral"

    if contains_any([
        "why",
        "how does",
        "internally",
        "under the hood",
        "in production",
        "trade off",
        "tradeoff",
        "pros and cons",
    ]) and has_technical_term:
        return "conceptual_deep"

    if contains_any(["what is", "what are", "explain", "define", "describe"]):
        return "conceptual_basic"

    if contains_any([
        "implement",
        "leetcode",
        "algorithm",
        "complexity",
        "binary",
        "array",
        "linked list",
        "tree",
        "graph",
    ]):
        return "coding"

    return "coding"


class InterviewOrchestrator:
    def __init__(self, llm_client: LlmClient) -> None:
        self.llm_client = llm_client
        self.default_prompt = ""
        self.default_mode = "auto"
        self.conversation_history: list[dict] = []
        self.resume_context: str = ""
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

    def load_resume(self, path: str) -> None:
        from icc.core.resume_loader import load_resume

        self.resume_context = load_resume(path)

    def _trim_history(self) -> None:
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = self.conversation_history[-MAX_HISTORY:]

    def _system_prompt_template(self) -> str:
        return (
            "You are a senior backend engineer answering in a real technical interview.\n"
            "Speak in formal, professional English as if talking to a hiring panel.\n"
            "Use first person (I). No markdown. No bullet points.\n"
            "No filler phrases like \"great question\", \"absolutely\", \"certainly\", or \"of course\".\n"
            "Do not open with a definition unless the question type is CONCEPTUAL_BASIC or CONCEPTUAL_DEEP.\n"
            "Do not invent any experience not listed in the candidate background.\n"
            "Include at least one concrete metric or number where it naturally fits.\n"
            "Target length: 45 to 60 seconds of spoken delivery (roughly 100 to 130 words).\n"
            "Do not pad the answer - stop when the structure is complete.\n\n"
            "## CODING\n"
            "Restate the problem and confirm key constraints. Explain your approach and the reasoning "
            "behind it. Write a clean Java implementation. State time and space complexity. Call out one "
            "edge case. Optionally mention one optimization or alternative.\n\n"
            "## SYSTEM_DESIGN\n"
            "Clarify functional requirements and scale. Describe the high-level architecture. Explain the "
            "role of key components. Describe the data flow. Identify bottlenecks and how you would scale. "
            "State one deliberate trade-off and why you made it.\n\n"
            "## BEHAVIORAL\n"
            "Briefly set the situation. State your task and responsibility. Describe your specific action "
            "and the decision you made. State the result with measurable impact if possible. Optionally "
            "reflect on what you learned.\n\n"
            "## CONCEPTUAL_BASIC\n"
            "Give a one-line definition. Explain how it works simply. State when you would use it. Name "
            "one limitation or gotcha.\n\n"
            "## CONCEPTUAL_DEEP\n"
            "Start with a simple explanation. Explain the internal mechanism. Describe a real-world failure "
            "case or production scenario. State the trade-offs.\n\n"
            "## DEEP_DIVE\n"
            "Set the context briefly. Describe what you built or owned. Explain the key technical decision "
            "you made. Describe a challenge or failure you hit. State the trade-off.\n\n"
            "## DEBUGGING\n"
            "Clarify the symptoms. Describe what metrics or logs you would check. Explain how you would "
            "isolate the problem. State your hypothesis. Describe how you would validate it. Describe the fix.\n\n"
            "## SCENARIO\n"
            "Clarify the situation and constraints. Describe your immediate mitigation. Explain how you "
            "would find the root cause. Describe the long-term fix. State what monitoring or prevention you "
            "would put in place.\n\n"
            "## FOLLOW_UP\n"
            "Directly address the follow-up without re-introducing context. Be concise. If it requires a "
            "different structure, apply the matching type above.\n\n"
            "Candidate background (use only what is listed, do not invent):\n"
            "{resume}"
        )

    def request_answer(
        self,
        prompt: str,
        mode: str,
        on_chunk: Callable[[str], None],
        on_complete: Callable[[], None],
        on_error: Callable[[str], None],
        force_qtype: str | None = None,
        images_b64: list[str] | None = None,
    ) -> None:
        self._request_started_at = perf_counter()
        has_images = images_b64 is not None and len(images_b64) > 0
        qtype = force_qtype or classify_question(prompt, has_images=has_images)
        logger.info("Detected question type: %s", qtype)
        worker = Thread(
            target=self._run_request,
            args=(prompt, qtype, on_chunk, on_complete, on_error, images_b64),
            daemon=True,
        )
        worker.start()

    def _run_request(
        self,
        prompt: str,
        qtype: str,
        on_chunk: Callable[[str], None],
        on_complete: Callable[[], None],
        on_error: Callable[[str], None],
        images_b64: list[str] | None = None,
    ) -> None:
        system_message, user_message = self.build_prompt(prompt=prompt, qtype=qtype)
        history = self.conversation_history if qtype == "follow_up" else []
        try:
            full_response: list[str] = []
            buffer = ""
            for chunk in self.llm_client.stream_answer(
                prompt=user_message,
                system=system_message,
                history=history,
                images_b64=images_b64,
                model="gpt-4.1-mini",
            ):
                self._debug_log("enqueue_chunk", f"chars={len(chunk)} text={chunk!r}")
                buffer += chunk
                if len(buffer) > 24:
                    on_chunk(buffer)
                    buffer = ""
                full_response.append(chunk)
            if buffer:
                on_chunk(buffer)
        except RuntimeError as exc:
            on_error(str(exc))
            return

        self.conversation_history.append(
            {"role": "user", "content": f"[{qtype.upper()}] {prompt}"}
        )
        self.conversation_history.append(
            {"role": "assistant", "content": "".join(full_response)}
        )
        if qtype == "follow_up":
            self._trim_history()
        else:
            self.conversation_history = self.conversation_history[-2:]
        on_complete()

    def build_prompt(self, prompt: str, qtype: str) -> tuple[str, str]:
        question = prompt.strip() or self.default_prompt
        resume = self.resume_context or "No resume provided."
        system_message = self._system_prompt_template().format(resume=resume)
        user_message = (
            f"Question type: {qtype}. Follow the {qtype} structure strictly.\n\n"
            f"Question: {question}"
        )
        return system_message, user_message
