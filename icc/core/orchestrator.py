from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from threading import Thread
from time import perf_counter
from typing import Callable

from icc.llm.client import LlmClient
from icc.core.session_logger import SessionLogWriter


MAX_HISTORY = 4
LLM_MAX_TOKENS = 3000
GROQ_MAX_OUTPUT_TOKENS = 1000
TYPE_MAX_TOKENS = {
    "CONCEPTUAL_BASIC": 400,
    "FOLLOW_UP": 300,
    "SYSTEM_DESIGN": 600,
    "DEEP_DIVE": 600,
    "BEHAVIORAL": 700,
    "CONCEPTUAL_DEEP": 600,
    "DEBUGGING_SCENARIO": 600,
    "CODING": 900,
}
CLASSIFIER_MODEL = "llama-3.1-8b-instant"
TYPE_CODE_MAP = {
    "BE": "BEHAVIORAL",
    "CO": "CODING",
    "CB": "CONCEPTUAL",
    "DD": "DEEP_DIVE",
    "DS": "DEBUGGING_SCENARIO",
    "FU": "FOLLOW_UP",
    "SD": "SYSTEM_DESIGN",
}
DEPTH_CODE_MAP = {
    "B": "basic",
    "P": "practical",
    "D": "deep",
}
VALID_DEPTHS = {"basic", "practical", "deep"}
TYPE_PATTERN = re.compile(
    r'\[(?:T|TYPES?)\s*:\s*([A-Z_]+)',
    re.IGNORECASE,
)
DEPTH_PATTERN = re.compile(
    r'\[(?:D|DEPTH)\s*:\s*(basic|practical|deep|[BPD])\b'
    r'|\b(DEEP|PRACTICAL|BASIC)\b(?!\s*_)',
    re.IGNORECASE,
)
CLASSIFIER_PROMPT = """You are a question classifier.
Respond with EXACTLY two tags and nothing else.
Format: [T:XX] [D:Y]

Do NOT output explanations.
Do NOT output full words like BEHAVIORAL or CONCEPTUAL.
Do NOT use [TYPE:...] or [DEPTH:...] format.
Output ONLY the two tags on a single line.

TYPE codes:
  BE = BEHAVIORAL
  CO = CODING
  CB = CONCEPTUAL (will be routed to BASIC or DEEP based on DEPTH)
  DD = DEEP_DIVE
  DS = DEBUGGING_SCENARIO
  FU = FOLLOW_UP
  SD = SYSTEM_DESIGN

DEPTH codes:
  B = basic (what is X, define X)
  P = practical (how do you handle X, how do you implement X, how would you)
  D = deep (design a system end-to-end)

Rules:
  - 'What is X' or 'Define X' → [D:B]
  - 'How do you handle/implement X' or 'How would you' → [D:P]
  - 'Design a system' or 'end-to-end' → [D:D]
  - 'Tell me about a time' → [T:BE]
  - 'Tell me about a system you built' → [T:DD] [D:P]
  - When in doubt between B and P, choose P

Example outputs:
  [T:BE] [D:P]
  [T:CB] [D:P]
  [T:SD] [D:D]"""
GROQ_HEAVY_MODEL = "openai/gpt-oss-120b"
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
        self.answer_mode: str = "grounded_senior"
        self.conversation_history: list[dict] = []
        self.resume_context: str = ""
        self.debug_stream = os.getenv("ICC_DEBUG_STREAM", "").strip() == "1"
        self._request_started_at: float | None = None
        self._session_request_id = 0
        self._session_log_writer = SessionLogWriter()
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

    def log_session_warning(self, message: str) -> None:
        self._session_log_writer.append_block(
            f"[WARNING | TIMESTAMP={datetime.now().isoformat(timespec='seconds')}]\n"
            f"{message}\n"
            "---\n"
        )

    def shutdown(self) -> None:
        self._session_log_writer.close(total_requests=self._session_request_id)

    def load_resume(self, path: str) -> None:
        from icc.core.resume_loader import load_resume

        self.resume_context = load_resume(path)

    def _trim_history(self) -> None:
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = self.conversation_history[-MAX_HISTORY:]

    def _strip_type_prefix(self, response_text: str) -> str:
        return re.sub(r"^\s*\[TYPE:[^\]]+\]\s*", "", response_text, count=1)

    def _append_session_log(
        self,
        request_id: int,
        question_type: str,
        transcript: str,
        response_text: str,
    ) -> None:
        block = (
            f"[REQUEST_ID={request_id} | TYPE={question_type} | "
            f"TIMESTAMP={datetime.now().isoformat(timespec='seconds')}]\n"
            f"TRANSCRIPT: {transcript}\n"
            f"RESPONSE: {self._strip_type_prefix(response_text)}\n"
            "---\n"
        )
        self._session_log_writer.append_block(block)

    def _extract_type(self, raw: str) -> str | None:
        for qt in QUESTION_TYPES:
            if re.search(r'\b' + qt + r'\b', raw, re.IGNORECASE):
                return qt
        return None

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
        print("[CLASSIFIER PROMPT]", CLASSIFIER_PROMPT)
        try:
            raw = self.llm_client.complete_text(
                system=CLASSIFIER_PROMPT,
                prompt=transcript,
                model=CLASSIFIER_MODEL,
                max_tokens=20,
                api_key=groq_api_key,
                base_url=self.llm_client.config.groq_base_url,
                timeout=CLASSIFIER_TIMEOUT_SECONDS,
            )
            # Layer 1: robust parse (handles [T:XX], [TYPE:XX], [TYPES:XX])
            type_match = TYPE_PATTERN.search(raw)
            depth_match = DEPTH_PATTERN.search(raw)
            t_code = type_match.group(1).upper() if type_match else None
            parsed_type = TYPE_CODE_MAP[t_code] if t_code in TYPE_CODE_MAP else None
            # Depth: bracket form (group 1) or bare-word form (group 2)
            if depth_match:
                val = (depth_match.group(1) or depth_match.group(2)).upper()
                parsed_depth = DEPTH_CODE_MAP.get(val) or val.lower()
                if parsed_depth not in VALID_DEPTHS:
                    parsed_depth = "basic"
            else:
                parsed_depth = "basic"
            # Layer 2: word boundary fallback (keep existing _extract_type)
            if not parsed_type:
                parsed_type = self._extract_type(raw) or "OTHER"
            # CONCEPTUAL depth routing
            if parsed_type == "CONCEPTUAL":
                question_type = "CONCEPTUAL_DEEP" if parsed_depth != "basic" \
                                else "CONCEPTUAL_BASIC"
            else:
                question_type = parsed_type
            print(f"[CLASSIFIER RAW] {raw!r}")
            print(f"[CLASSIFIER PARSED] type={question_type} depth={parsed_depth}")
            elapsed_ms = (perf_counter() - started_at) * 1000
            logger.info("Classified as %s depth=%s (%.0f ms)", question_type, parsed_depth, elapsed_ms)
            return question_type
        except RuntimeError as exc:
            logger.warning("Groq classify failed, using OTHER: %s", exc)
            return OTHER_TYPE

    def _select_model(self, question_type: str, has_image: bool) -> str:
        if has_image:
            return OPENAI_VISION_MODEL
        return GROQ_HEAVY_MODEL

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
            buffer += chunk
            if len(buffer) > 24:
                buffer = self._flush_buffer(buffer, on_chunk)
            full_response.append(chunk)
        return buffer

    def _system_prompt_template(
        self,
        answer_mode: str = "grounded_senior",
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
            "Do NOT invent exotic failure scenarios "
            "(e.g., rare distributed partition edge cases)\n"
            "unless clearly framed as hypothetical with 'In theory...' or\n"
            "'A hypothetical risk is...'.\n"
            "Prefer failures that are common in backend financial systems\n"
            "and can be tied to real observability signals "
            "(Prometheus metrics, GC logs, slow query logs).\n\n"
            "For BEHAVIORAL: name the specific project from "
            "resume in sentence 1. Include one measurable result.\n"
            "State one lesson or tradeoff explicitly.\n\n"
            "Do NOT place trade-offs or failure modes as an afterthought.\n"
            "Integrate them naturally into the explanation.\n"
        )
        if question_type in {
            "SYSTEM_DESIGN",
            "DEBUGGING_SCENARIO",
            "DEEP_DIVE",
            "CONCEPTUAL_DEEP",
            "BEHAVIORAL",
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
                "For BEHAVIORAL: name the specific project from "
                "resume in sentence 1. Include one measurable result.\n"
                "State one lesson or tradeoff explicitly.\n\n"
                "Do NOT place them at the end as an afterthought. "
                "Integrate them naturally into your explanation.\n\n"
            )
            if answer_mode == "grounded_senior":
                addon = grounded_senior_addon
            elif answer_mode == "senior":
                addon = senior_addon
        return addon + (
            "SECTION 1 - ROLE & CONTEXT\n"
            "You are a senior backend engineer answering questions in a real technical interview.\n"
            "Keep the answer grounded in the candidate background and prefer specific projects and metrics over generic explanations.\n"
            "Candidate background:\n"
            "{resume}\n\n"
            "Use natural spoken English.\n"
            "Use first person.\n"
            "Do not invent experience not listed in the candidate background.\n"
            "Include a concrete number when it naturally fits.\n\n"
            "SECTION 2 - REASONING INSTRUCTION\n"
            "Before answering, internally identify: (1) question type, (2) what the interviewer is testing, (3) most relevant resume evidence.\n"
            "Do not output this reasoning.\n\n"
            "SECTION 4 - OUTPUT CONSTRAINTS\n"
            "BEHAVIORAL: stay within 250 tokens.\n"
            "SYSTEM_DESIGN: stay within 500 tokens, layer overview first, 2 tradeoffs at the end.\n"
            "FOLLOW_UP: stay within 100 tokens.\n"
            "CONCEPTUAL_DEEP: stay within 300 tokens.\n"
            "DEEP_DIVE: stay within 450 tokens.\n"
            "DEBUGGING_SCENARIO: stay within 280 tokens.\n"
            "Never use markdown formatting: no bold, no bullet points, no emoji, and no numbered lists.\n"
            "Answers must read as natural spoken English only.\n"
        )

    def request_answer(
        self,
        prompt: str,
        mode: str,
        on_chunk: Callable[[str], None],
        on_complete: Callable[[str], None],
        on_error: Callable[[str], None],
        images_b64: list[str] | None = None,
        answer_mode: str = "grounded_senior",
    ) -> None:
        self.answer_mode = answer_mode
        self._request_started_at = perf_counter()
        self._session_request_id += 1
        request_id = self._session_request_id
        worker = Thread(
            target=self._run_request,
            args=(request_id, prompt, on_chunk, on_complete, on_error, images_b64, answer_mode),
            daemon=True,
        )
        worker.start()

    def _run_request(
        self,
        request_id: int,
        prompt: str,
        on_chunk: Callable[[str], None],
        on_complete: Callable[[str], None],
        on_error: Callable[[str], None],
        images_b64: list[str] | None = None,
        answer_mode: str = "grounded_senior",
    ) -> None:
        is_follow_up = "[This is a follow-up" in prompt
        history = self.conversation_history
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
        max_tokens = TYPE_MAX_TOKENS.get(question_type, LLM_MAX_TOKENS)
        if groq_enabled and not use_openai_direct:
            max_tokens = min(max_tokens, GROQ_MAX_OUTPUT_TOKENS)
        logger.info(
            "Routing to model: %s",
            selected_model if groq_enabled else OPENAI_FALLBACK_MODEL,
        )
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
                "max_tokens": max_tokens,
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
                fallback_kwargs: dict[str, object] = {
                    "prompt": user_message,
                    "system": system_message,
                    "history": history,
                    "images_b64": images_b64,
                    "model": OPENAI_FALLBACK_MODEL,
                    "max_tokens": max_tokens,
                }
                fallback_stream = self.llm_client.stream_answer(**fallback_kwargs)
                buffer = self._consume_stream(fallback_stream, on_chunk, full_response, buffer)
            buffer = self._flush_buffer(buffer, on_chunk)
        except RuntimeError as exc:
            self._append_session_log(
                request_id=request_id,
                question_type=question_type,
                transcript=question,
                response_text="".join(full_response) if "full_response" in locals() else str(exc),
            )
            on_error(str(exc))
            return

        response_text = "".join(full_response)
        self._append_session_log(
            request_id=request_id,
            question_type=question_type,
            transcript=question,
            response_text=response_text,
        )
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append(
            {"role": "assistant", "content": response_text}
        )
        if is_follow_up:
            self._trim_history()
        else:
            self.conversation_history = self.conversation_history[-2:]
        on_complete(response_text)

    def build_prompt(
        self,
        prompt: str,
        answer_mode: str = "grounded_senior",
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
