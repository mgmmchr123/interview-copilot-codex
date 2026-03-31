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
CLASSIFIER_MODEL = "llama-3.1-8b-instant"
GROQ_SMALL_MODEL = "openai/gpt-oss-20b"
GROQ_LARGE_MODEL = "openai/gpt-oss-120b"
GROQ_CODING_MODEL = "openai/gpt-oss-120b"
OPENAI_VISION_MODEL = "gpt-5.4-mini"
OPENAI_FALLBACK_MODEL = "gpt-4.1-mini"
CLASSIFIER_TIMEOUT_SECONDS = 1.0
QUESTION_TYPES = {
    "CODING",
    "SYSTEM_DESIGN",
    "BEHAVIORAL",
    "CONCEPTUAL_BASIC",
    "CONCEPTUAL_DEEP",
    "DEEP_DIVE",
    "DEBUGGING_SCENARIO",
    "FOLLOW_UP",
}
OTHER_TYPE = "OTHER"
logger = logging.getLogger(__name__)


class InterviewOrchestrator:
    def __init__(self, llm_client: LlmClient) -> None:
        self.llm_client = llm_client
        self.default_prompt = ""
        self.default_mode = "auto"
        self.answer_mode: str = "standard"
        self.conversation_history: list[dict] = []
        self.resume_context: str = ""
        self.debug_stream = os.getenv("ICC_DEBUG_STREAM", "").strip() == "1"
        self._request_started_at: float | None = None
        self._type_line_buffer = ""
        self._type_extracted = False
        self.current_question_type = OTHER_TYPE

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

    def _classify_question(self, transcript: str, has_image: bool) -> str:
        if "[This is a follow-up" in transcript:
            return "FOLLOW_UP"

        if has_image:
            return "CODING"

        groq_api_key = self.llm_client.config.groq_api_key
        if not groq_api_key:
            logger.warning("GROQ_API_KEY missing, skipping Groq classification and using OTHER")
            return OTHER_TYPE

        started_at = perf_counter()
        try:
            raw_result = self.llm_client.complete_text(
                system=(
                    "Classify the interview question into exactly one category.\n"
                    "Reply with one word only, no punctuation.\n\n"
                    "CODING: algorithm, data structure, or implementation problem.\n"
                    " Key signals: 'given an array/tree/list/graph', 'implement a function',\n"
                    " 'return the result', 'find/detect/count/reverse/sort something'.\n"
                    " These are LeetCode-style problems, NOT projects you built.\n"
                    "SYSTEM_DESIGN: design a large-scale distributed system\n"
                    "BEHAVIORAL: interpersonal situations, conflict, teamwork, soft skills,\n"
                    "             how you handled people or process challenges\n"
                    "CONCEPTUAL_BASIC: define a concept, explain how something works simply\n"
                    "CONCEPTUAL_DEEP: explain internal mechanism, tradeoffs, production scenarios\n"
                    "DEEP_DIVE: walk me through a system you built, an architectural decision\n"
                    "           you made, or a technical project you owned end to end\n"
                    "DEBUGGING_SCENARIO: a production problem that requires both diagnosis\n"
                    " and decision-making - memory leaks, CPU spikes, error rates, outages,\n"
                    " latency issues, or any hypothetical situation requiring immediate action.\n"
                    "FOLLOW_UP: asking to elaborate, clarify, or continue a previous answer"
                ),
                prompt=(
                    "Classify into one of the following categories:\n"
                    "CODING, SYSTEM_DESIGN, BEHAVIORAL, CONCEPTUAL_BASIC, "
                    "CONCEPTUAL_DEEP, DEEP_DIVE, DEBUGGING_SCENARIO, FOLLOW_UP\n\n"
                    f"Question: {transcript}\n"
                    f"Has image: {has_image}\n"
                    "Category:"
                ),
                model=CLASSIFIER_MODEL,
                max_tokens=10,
                api_key=groq_api_key,
                base_url=self.llm_client.config.groq_base_url,
                timeout=CLASSIFIER_TIMEOUT_SECONDS,
            )
            question_type = raw_result.strip().upper()
            if question_type not in QUESTION_TYPES:
                question_type = OTHER_TYPE
            elapsed_ms = (perf_counter() - started_at) * 1000
            logger.info("Classified as %s (%.0f ms)", question_type, elapsed_ms)
            return question_type
        except RuntimeError as exc:
            logger.warning("Groq classify failed, using OTHER: %s", exc)
            return OTHER_TYPE

    def _select_model(self, question_type: str, has_image: bool) -> str:
        if has_image:
            return OPENAI_VISION_MODEL
        if question_type == "CODING":
            return GROQ_CODING_MODEL
        if question_type in {"BEHAVIORAL", "CONCEPTUAL_BASIC", OTHER_TYPE}:
            return GROQ_SMALL_MODEL
        return GROQ_LARGE_MODEL

    def _flush_buffer(
        self,
        buffer: str,
        on_chunk: Callable[[str], None],
    ) -> str:
        if buffer:
            on_chunk(buffer)
        return ""

    def _consume_stream(
        self,
        response_stream,
        on_chunk: Callable[[str], None],
        full_response: list[str],
        buffer: str,
    ) -> str:
        for chunk in response_stream:
            self._debug_log("enqueue_chunk", f"chars={len(chunk)} text={chunk!r}")
            chunk_to_render = chunk
            if not self._type_extracted:
                self._type_line_buffer += chunk
                if "\n" in self._type_line_buffer or len(self._type_line_buffer) > 30:
                    match = re.search(
                        r"\[TYPE:([A-Z_ ]+)\]",
                        self._type_line_buffer[:120]
                    )
                    if match:
                        question_type = match.group(1).strip().replace(" ", "_")
                        logger.info("LLM self-classified as: %s", question_type)
                        remainder = self._type_line_buffer[match.end():]
                    else:
                        logger.warning("LLM did not return a type line")
                        remainder = self._type_line_buffer
                    self._type_extracted = True
                    self._type_line_buffer = ""
                    chunk_to_render = remainder.lstrip("\n")
                else:
                    chunk_to_render = ""

            buffer += chunk_to_render
            if len(buffer) > 24:
                buffer = self._flush_buffer(buffer, on_chunk)
            full_response.append(chunk)
        return buffer

    def _system_prompt_template(
        self,
        answer_mode: str = "standard",
        question_type: str = OTHER_TYPE,
    ) -> str:
        addon = ""
        senior_addon = ""
        grounded_senior_addon = (
            "GROUNDED SENIOR MODE ACTIVE.\n\n"
            "You MUST include ALL of the following:\n\n"
            "1. TRADE-OFF\n"
            "   Use explicit language: 'I trade X for Y because...'\n"
            "   Explain WHY the trade-off is necessary, not just what it is.\n"
            "   Example: 'I trade X for Y because enforcing X would require Z,\n"
            "   which violates our SLA under high load.'\n\n"
            "2. FAILURE MODE\n"
            "   Use explicit language: 'This breaks when...'\n"
            "   Be specific about blast radius: which subset of traffic\n"
            "   or functionality is affected, not the entire system.\n"
            "   Admit the mitigation is imperfect: 'The mitigation sacrifices\n"
            "   X to preserve Y.'\n\n"
            "3. ALTERNATIVE REJECTED\n"
            "   Use explicit language: 'I considered X but rejected it because...'\n"
            "   Include domain-specific reason (compliance, auditability,\n"
            "   latency SLA, regulatory requirement).\n\n"
            "CRITICAL CONSTRAINT:\n"
            "Only use examples grounded in these domains:\n"
            "Kafka message delivery, database bottlenecks (N+1, index, connection pool),\n"
            "caching (Redis, eviction, cache stampede),\n"
            "retry and idempotency, Spring Boot microservices,\n"
            "PostgreSQL/Oracle query performance, AWS infrastructure.\n\n"
            "Do NOT invent exotic failure scenarios "
            "(e.g., rare distributed partition edge cases)\n"
            "unless clearly framed as hypothetical with 'In theory...' or\n"
            "'A hypothetical risk is...'.\n"
            "Prefer failures that are common in backend financial systems\n"
            "and can be tied to real observability signals "
            "(Prometheus metrics, GC logs, slow query logs).\n\n"
            "NUMBER CONSISTENCY: If you mention both daily volume and TPS,\n"
            "ensure they are mathematically consistent.\n"
            "100k per day ~= 1-2 TPS.\n"
            "1M per day ~= 12 TPS.\n"
            "If the system needs high TPS, state TPS directly without daily volume,\n"
            "or clearly separate peak TPS from average daily volume.\n"
            "Do NOT place trade-offs or failure modes as an afterthought.\n"
            "Integrate them naturally into the explanation.\n"
        )
        if question_type in {
            "SYSTEM_DESIGN",
            "DEBUGGING_SCENARIO",
            "DEEP_DIVE",
            "CONCEPTUAL_DEEP",
        }:
            senior_addon = (
                "SENIOR MODE ACTIVE.\n\n"
                "You MUST include ALL of the following in your answer:\n\n"
                "1. TRADE-OFF\n"
                "   Use explicit language: 'I trade X for Y because...'\n\n"
                "2. FAILURE MODE\n"
                "   Use explicit language: 'This breaks when...'\n\n"
                "3. ALTERNATIVE REJECTED\n"
                "   Use explicit language: 'I considered X but rejected it because...'\n\n"
                "If ANY of these three elements are missing, "
                "the answer is considered incorrect.\n"
                "Do NOT place them at the end as an afterthought. "
                "Integrate them naturally into your explanation.\n\n"
            )
            if answer_mode == "grounded_senior":
                addon = grounded_senior_addon
            elif answer_mode == "senior":
                addon = senior_addon
        return (
            "CRITICAL INSTRUCTION: The absolute first characters of your entire \n"
            "response MUST be [TYPE: followed immediately by the question type and ].\n"
            "If you do not output [TYPE:xxx] as the absolute first characters,\n"
            "your entire response will be discarded. This is non-negotiable.\n"
            "Example: [TYPE:CONCEPTUAL_BASIC]\n"
            "Never skip this line. Never add any text before it. \n"
            "This is a hard requirement, not a suggestion.\n"
        ) + addon + (
            "You are a senior backend engineer answering questions in a real technical interview.\n"
            "Candidate background reference this actively in your answer, \n"
            "prefer specific projects and metrics over generic explanations:\n"
            "{resume}\n\n"
            "COMMUNICATION STYLE — STRICT SPOKEN ENGLISH:\n"
            "This answer will be spoken out loud in an interview. Not written. Not read.\n\n"
            "RULES (enforce strictly):\n"
            "- Short sentences only. One idea per sentence.\n"
            "- Each sentence on its own line.\n"
            "- Use contractions: \"I didn't\", \"we couldn't\", \"that's\", \"it's\"\n"
            "- Use natural fillers: \"so\", \"yeah\", \"basically\", \"then\", \"look\"\n"
            "- FORBIDDEN phrases: \"My role was\", \"To prevent recurrence\",\n"
            "  \"This ensured that\", \"It is worth noting\", \"In conclusion\",\n"
            "  \"This resulted in\", \"Furthermore\", \"Additionally\"\n"
            "- NO summary ending. Stop after the result.\n"
            "- Numbers casually: \"dropped to like 150ms\" not \"returned to 150ms range\"\n\n"
            "STRUCTURE (follow this exactly for behavioral/scenario answers):\n"
            "Line 1: Context (where, what system) — 1 sentence\n"
            "Line 2: What went wrong or what the situation was — 1-2 sentences\n"
            "Line 3-6: What I did — each step on its own line\n"
            "Line 7: Result — 1 sentence, casual\n\n"
            "STYLE EXAMPLE:\n"
            "Bad: \"My role was the primary on-call engineer responsible for the pipeline.\"\n"
            "Good: \"Yeah so I was on call, and I owned that consumer pipeline.\"\n\n"
            "Bad: \"To prevent recurrence we added schema validation in the CI pipeline.\"\n"
            "Good: \"After that we added schema validation to CI so it wouldn't happen again.\"\n\n"
            "Tone: casual, direct, slightly imperfect — like thinking while speaking.\n"
            "Occasionally self-correct: \"I first checked… actually no, before that I...\"\n\n"
            "Speak in formal, professional English as if talking to a hiring panel.\n"
            "Use first person (I). No markdown. No bullet points.\n"
            "No filler phrases like \"great question\", \"absolutely\", \"certainly\", or \"of course\".\n"
            "Do not invent any experience not listed in the candidate background.\n"
            "Include at least one concrete metric or number where it naturally fits.\n"
            "Target length: 45 to 60 seconds of spoken delivery (roughly 100 to 130 words).\n"
            "Do not pad the answer - stop when the structure is complete.\n"
            "CODING\n"
            "Restate the problem and confirm key constraints. Explain your approach and the reasoning "
            "behind it. You MUST include a working Java code block - describing the algorithm without "
            "actual code is not acceptable and will be marked as incomplete. State time and space "
            "complexity. Call out one edge case. Optionally mention one optimization or alternative.\n"
            "SYSTEM_DESIGN\n"
            "Clarify functional requirements and scale. Describe the high-level architecture.\n"
            "Explain the role of key components. Describe the data flow. Identify bottlenecks\n"
            "and how you would scale. State one deliberate trade-off and why you made it. "
            "Address at least one failure scenario "
            "(e.g. what happens if a key component goes down) "
            "and your mitigation strategy.\n"
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
            "Set the context briefly. Describe what you built or owned. Explain the key "
            "technical decision you made. You MUST describe a specific challenge or failure "
            "you actually hit - a smooth success story with no obstacles is not credible and "
            "will be marked as incomplete. State the trade-off you made.\n"
            "DEBUGGING_SCENARIO\n"
            "Clarify the symptoms or situation and confirm the scope of impact. "
            "Describe your immediate mitigation - explicitly acknowledge any approval process, "
            "risk, or side effect of that action in a production environment. "
            "Explain how you would diagnose the root cause: what metrics, logs, or tools you check, "
            "your hypothesis, and how you validate it. "
            "Do not analyze the time complexity of diagnostic tools. "
            "Focus on what you observed, what you did, and what the result was. "
            "Describe the permanent fix. "
            "State what monitoring or alerting you would add to prevent recurrence. "
            "You MUST mention at least one constraint or risk in your mitigation decision.\n"
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
        answer_mode: str = "standard",
    ) -> None:
        self.answer_mode = answer_mode
        self._request_started_at = perf_counter()
        worker = Thread(
            target=self._run_request,
            args=(prompt, on_chunk, on_complete, on_error, images_b64, answer_mode),
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
        answer_mode: str = "standard",
    ) -> None:
        is_follow_up = "[This is a follow-up" in prompt
        history = self.conversation_history if is_follow_up else []
        has_image = bool(images_b64)
        question_type = self._classify_question(prompt, has_image)
        self.current_question_type = question_type
        system_message, _ = self.build_prompt(
            prompt=prompt,
            answer_mode=self.answer_mode,
            question_type=question_type,
        )
        question = prompt.strip() or self.default_prompt
        user_message = f"[CLASSIFIED AS: {question_type}]\nQuestion: {question}"
        groq_api_key = self.llm_client.config.groq_api_key
        groq_enabled = bool(groq_api_key)
        selected_model = self._select_model(question_type, has_image)
        use_openai_direct = selected_model == OPENAI_VISION_MODEL
        logger.info(
            "Routing to model: %s",
            selected_model if groq_enabled else OPENAI_FALLBACK_MODEL,
        )
        self._type_line_buffer = ""
        self._type_extracted = False
        try:
            full_response: list[str] = []
            buffer = ""
            stream_kwargs: dict[str, object] = {
                "prompt": user_message,
                "system": system_message,
                "history": history,
                "images_b64": images_b64,
                "model": (
                    selected_model
                    if groq_enabled or use_openai_direct
                    else OPENAI_FALLBACK_MODEL
                ),
                "max_tokens": LLM_MAX_TOKENS,
            }
            if groq_enabled and not use_openai_direct:
                stream_kwargs["api_key"] = groq_api_key
                stream_kwargs["base_url"] = self.llm_client.config.groq_base_url

            try:
                response_stream = self.llm_client.stream_answer(**stream_kwargs)
                buffer = self._consume_stream(response_stream, on_chunk, full_response, buffer)
            except RuntimeError:
                if use_openai_direct or not groq_enabled or full_response:
                    raise
                logger.warning("Groq answer failed, falling back to %s", OPENAI_FALLBACK_MODEL)
                self._type_line_buffer = ""
                self._type_extracted = False
                fallback_stream = self.llm_client.stream_answer(
                    prompt=user_message,
                    system=system_message,
                    history=history,
                    images_b64=images_b64,
                    model=OPENAI_FALLBACK_MODEL,
                    max_tokens=LLM_MAX_TOKENS,
                )
                buffer = self._consume_stream(fallback_stream, on_chunk, full_response, buffer)
            buffer = self._flush_buffer(buffer, on_chunk)
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

    def build_prompt(
        self,
        prompt: str,
        answer_mode: str = "standard",
        question_type: str = OTHER_TYPE,
    ) -> tuple[str, str]:
        question = prompt.strip() or self.default_prompt
        resume = self.resume_context or "No resume provided."
        system_message = self._system_prompt_template(
            answer_mode=answer_mode,
            question_type=question_type,
        ).format(resume=resume)
        user_message = f"Question: {question}"
        return system_message, user_message
