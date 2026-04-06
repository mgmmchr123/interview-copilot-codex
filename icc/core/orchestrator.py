from __future__ import annotations

import logging
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from threading import Thread
from time import perf_counter
from typing import Callable

from config.interview_context import CONTEXT_PROFILES
from icc.core.context_cards import select_cards
from icc.core.story_bank import (
    find_story, is_experience_followup, get_snippet, get_summary
)
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
GROQ_HEAVY_MODEL = "llama-3.3-70b-versatile"
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
TYPE_PATTERNS = [
    ("DEBUGGING_SCENARIO", r"\bDEBUGGING_SCENARIO\b"),
    ("SYSTEM_DESIGN",      r"\bSYSTEM_DESIGN\b"),
    ("BEHAVIORAL",         r"\bBEHAVIORAL\b"),
    ("CODING",             r"\bCODING\b"),
    ("FOLLOW_UP",          r"\bFOLLOW_UP\b"),
    ("DEEP_DIVE",          r"\bDEEP_DIVE\b"),
    ("CONCEPTUAL",         r"\bCONCEPTUAL\b"),
]
TYPE_PATTERN = re.compile(
    r'\[(?:T|TYPE[S]?)\s*:\s*([A-Z_]+)',
    re.IGNORECASE
)
DEPTH_PATTERN = re.compile(
    r'\[(?:D|DEPTH)\s*:\s*(basic|practical|deep|[BPD])\b'
    r'|\b(DEEP|PRACTICAL|BASIC)\b(?!\s*_)',
    re.IGNORECASE
)
TYPE_CODE_MAP = {
    "BE": "BEHAVIORAL",
    "BEHAVIORAL": "BEHAVIORAL",
    "SD": "SYSTEM_DESIGN",
    "SYSTEM_DESIGN": "SYSTEM_DESIGN",
    "DD": "DEEP_DIVE",
    "DEEP_DIVE": "DEEP_DIVE",
    "DB": "DEBUGGING_SCENARIO",
    "DEBUGGING_SCENARIO": "DEBUGGING_SCENARIO",
    "CO": "CONCEPTUAL",
    "CONCEPTUAL": "CONCEPTUAL",
    "FU": "FOLLOW_UP",
    "FOLLOW_UP": "FOLLOW_UP",
    "CD": "CODING",
    "CODING": "CODING",
}
DEPTH_CODE_MAP = {
    "B": "basic",
    "P": "practical",
    "D": "deep",
    "BASIC": "basic",
    "PRACTICAL": "practical",
    "DEEP": "deep",
}
VALID_DEPTHS = {"basic", "practical", "deep"}

BEHAVIORAL_PATTERNS = [
    "tell me about a time",
    "describe a time",
    "walk me through a time",
    "give me an example",
    "can you give me an example",
    "can you walk me through",
    "share an example",
    "what's an example",
    "have you ever",
]

BEHAVIORAL_VERBS = [
    "influence", "influenced", "convince", "convinced",
    "drive", "drove", "push", "pushed", "advocate", "advocated",
    "disagree", "disagreed", "resolve", "resolved",
    "conflict", "handle conflict",
    "lead", "led", "manage", "managed",
    "mentor", "mentored", "guide", "guided",
    "decide", "decided", "choose", "chose",
    "prioritize", "prioritized",
    "fail", "failed", "mistake", "wrong",
    "learned", "lesson",
    "handle pressure", "multiple priorities",
    "refuse", "refused", "resist", "resisted", "pushback", "pushed back",
]

FOLLOWUP_KEYWORDS = [
    "what if", "how do you handle", "what happens if",
    "how do you ensure", "crash", "failure", "retry",
    "consistency", "still correct", "atomicity",
    "guarantee", "ensure", "correctness", "why safe",
]

HARD_SWITCH_SIGNALS = [
    "let's switch", "new question", "different topic",
    "unrelated", "move on to", "tell me about",
    "describe a time", "walk me through a time",
    "give me an example", "different project",
    "another system", "at your previous",
]

MAX_DEPTH = 3


def is_followup_by_keyword(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in FOLLOWUP_KEYWORDS)


def introduces_new_topic(q: str, topic_keywords: set) -> bool:
    ql = q.lower()
    if any(s in ql for s in HARD_SWITCH_SIGNALS):
        return True
    tokens = set(w for w in ql.split() if len(w) > 4)
    return len(tokens) > 0 and len(tokens & topic_keywords) == 0


def is_followup_by_context(q: str, topic_keywords: set) -> bool:
    if not topic_keywords:
        return False  # No topic context yet — do not guess
    tokens = q.split()
    return len(tokens) < 20 and not introduces_new_topic(q, topic_keywords)


def is_followup(q: str, phase_state, topic_keywords: set) -> bool:
    # Allow keyword detection even when phase_state is None
    if phase_state is not None and not phase_state.active:
        return False
    return is_followup_by_keyword(q) or is_followup_by_context(q, topic_keywords)


def map_to_depth(q: str, current_depth: int) -> int:
    ql = q.lower()
    matched = []
    if any(x in ql for x in ["ensure", "guarantee", "consistency"]):
        matched.append(1)
    if any(x in ql for x in ["what if", "failure", "crash", "retry"]):
        matched.append(2)
    if any(x in ql for x in ["still correct", "why safe", "correctness", "atomicity"]):
        matched.append(3)
    if not matched:
        return min(current_depth + 1, MAX_DEPTH)
    return min(max(current_depth, max(matched)), MAX_DEPTH)


def build_depth_block(depth: int) -> str:
    blocks = []
    if depth >= 1:
        blocks.append(
            "GUARANTEE REQUIRED — HIGHEST PRIORITY INSTRUCTION:\n"
            "This requirement overrides ALL other constraints, including any "
            "instruction about not introducing new details.\n\n"
            "Your FIRST sentence MUST be exactly in this form:\n"
            "'I guarantee [X], but I do NOT guarantee [Y].'\n\n"
            "You MUST explicitly state both sides:\n"
            "- what is guaranteed\n"
            "- what is NOT guaranteed\n\n"
            "You are allowed to introduce standard distributed systems terms "
            "(e.g., at-least-once, exactly-once, idempotency) even if they "
            "are not explicitly mentioned in the story.\n\n"
            "Do NOT skip this. Do NOT paraphrase. Do NOT delay it.\n"
            "If this sentence is missing, the answer is invalid."
        )
    if depth >= 2:
        blocks.append(
            "FAILURE WALKTHROUGH REQUIRED: Include one concrete failure scenario. "
            "Structure: what fails → what the system does → why it is still safe."
        )
    if depth >= 3:
        blocks.append(
            "INVARIANT PROOF REQUIRED: Explicitly state why the system remains "
            "correct after the failure. Then name one concrete trade-off accepted "
            "(latency, ops overhead, or debug complexity — not 'complexity increases')."
        )
    return "\n".join(blocks)


CLASSIFIER_PROMPT = (
    "Classify the interview question into TYPE and DEPTH.\n\n"
    "TYPE (choose exactly one):\n"
    "- BEHAVIORAL: interpersonal situations, conflict, teamwork, soft skills,\n"
    "              how you handled people or process challenges\n"
    "- CODING: algorithm, data structure, or implementation problem.\n"
    "  Key signals: 'given an array/tree/list/graph', 'implement a function',\n"
    "  'return the result', 'find/detect/count/reverse/sort something'.\n"
    "  These are LeetCode-style problems, NOT projects you built.\n"
    "- CONCEPTUAL: define a concept, explain how something works,\n"
    "              internal mechanisms, or production usage\n"
    "- SYSTEM_DESIGN: design a large-scale distributed system end to end\n"
    "- DEEP_DIVE: walk me through a system you built, an architectural decision\n"
    "             you made, or a technical project you owned end to end\n"
    "- DEBUGGING_SCENARIO: a production problem requiring diagnosis and\n"
    "  decision-making, memory leaks, CPU spikes, error rates, outages,\n"
    "  latency issues, or any hypothetical situation requiring immediate action\n"
    "- FOLLOW_UP: asking to elaborate, clarify, or continue a previous answer\n\n"
    "DEPTH (choose exactly one):\n"
    "- basic      (definition or simple explanation)\n"
    "- practical  (how it works in real systems)\n"
    "- deep       (full system design or architecture)\n\n"
    "Rules:\n"
    "- 'What is X' -> CONCEPTUAL + basic\n"
    "- 'How do you handle / implement X' -> CONCEPTUAL + practical\n"
    "- 'Design a system that does X end-to-end' -> SYSTEM_DESIGN + deep\n\n"
    "DEPTH rules:\n"
    "- 'What is X' or 'Define X' -> basic\n"
    "- 'How do you handle X' or 'How do you implement X' -> practical\n"
    "- 'Design a system that does X end-to-end' -> deep\n"
    "- 'Tell me about a system you built' -> practical\n"
    "- 'Walk me through...' -> practical\n"
    "When in doubt between basic and practical, choose practical.\n\n"
    "Output format: [TYPE:xxx] [DEPTH:yyy]\n"
    "Use exact TYPE names from the list above."
)


def _extract_type(raw: str) -> str | None:
    for label, pattern in TYPE_PATTERNS:
        if re.search(pattern, raw, re.IGNORECASE):
            return label
    return None


def parse_classifier_output(raw_result: str) -> tuple[str, str]:
    type_match = TYPE_PATTERN.search(raw_result)
    depth_match = DEPTH_PATTERN.search(raw_result)

    if type_match:
        code = type_match.group(1).upper()
        parsed_type = TYPE_CODE_MAP.get(code) or \
                      (code if code in QUESTION_TYPES else None)
    else:
        parsed_type = _extract_type(raw_result)

    if depth_match:
        val = (depth_match.group(1) or depth_match.group(2)).upper()
        parsed_depth = DEPTH_CODE_MAP.get(val) or val.lower()
        if parsed_depth not in VALID_DEPTHS:
            parsed_depth = "basic"
    else:
        parsed_depth = "basic"

    if parsed_type == "CONCEPTUAL":
        question_type = "CONCEPTUAL_DEEP" if parsed_depth != "basic" \
                        else "CONCEPTUAL_BASIC"
    elif parsed_type in QUESTION_TYPES:
        question_type = parsed_type
    else:
        question_type = OTHER_TYPE

    return question_type, parsed_depth


ADVANCE_PATTERNS = [
    r"^(ok|okay|yes|sure|go ahead|continue|proceed|next|next phase)",
    r"^(\u597d|\u7ee7\u7eed|\u4e0b\u4e00\u6b65|\u53ef\u4ee5|\u884c|\u660e\u767d\u4e86|\u5f00\u59cb|\u8fdb\u5165\u4e0b\u4e00)",
    r"^(got it|sounds good|that works|let'?s (go|continue|proceed))",
]
TERMINAL_PHASES = {"WRAP", "POSTMORTEM", "IMPACT_WRAP"}
logger = logging.getLogger(__name__)
DEFAULT_INTERVIEW_CONTEXT = "growth_tech"
TRADEOFF_STYLES = [
    "I chose this approach because...",
    "The reason I went with this is...",
    "One trade-off here is...",
    "This works well, but the downside is...",
    "The main limitation is...",
    "I went with X over Y because...",
]

QUESTION_CLASS_MAP = {
    "SYSTEM_DESIGN": "A",
    "DEBUGGING_SCENARIO": "A",
    "DEEP_DIVE": "A",
    "BEHAVIORAL": "B",
    "CONCEPTUAL_DEEP": "B",
    "CODING": "B",
    "OTHER": "B",
    "CONCEPTUAL_BASIC": "C",
    "FOLLOW_UP": "C",
}

PHASE_SEQUENCES = {
    "SYSTEM_DESIGN": ["CLARIFY", "HLD", "DEEP_DIVE", "SCALE_RELIABILITY", "WRAP"],
    "DEBUGGING_SCENARIO": ["TRIAGE", "MITIGATE", "INVESTIGATE", "FIX", "POSTMORTEM"],
    "DEEP_DIVE": ["PICK_CONTEXT", "ARCH_OVERVIEW", "DECISIONS", "EXECUTION", "IMPACT_WRAP"],
}


@dataclass
class PhaseState:
    question_type: str
    phase: str
    phase_version: int
    active: bool
    awaiting_user: bool
    slots: dict
    notes: dict

    @classmethod
    def start(cls, question_type: str) -> "PhaseState":
        sequence = PHASE_SEQUENCES[question_type]
        return cls(
            question_type=question_type,
            phase=sequence[0],
            phase_version=0,
            active=True,
            awaiting_user=True,
            slots={},
            notes={},
        )

    @classmethod
    def advance(cls, current: "PhaseState") -> "PhaseState":
        logger.info(
            "PhaseState.advance called: phase=%s active=%s v=%s",
            current.phase,
            current.active,
            current.phase_version,
        )
        if current.phase in TERMINAL_PHASES:
            logger.info(
                "PhaseState.advance: terminal phase %s → active=False",
                current.phase,
            )
            from dataclasses import replace

            return replace(current, active=False)

        sequence = PHASE_SEQUENCES[current.question_type]
        current_index = sequence.index(current.phase)
        if current_index >= len(sequence) - 1:
            return cls(
                question_type=current.question_type,
                phase=current.phase,
                phase_version=current.phase_version + 1,
                active=False,
                awaiting_user=False,
                slots=current.slots,
                notes=current.notes,
            )

        next_phase = sequence[current_index + 1]
        if next_phase in TERMINAL_PHASES:
            pass

        return cls(
            question_type=current.question_type,
            phase=next_phase,
            phase_version=current.phase_version + 1,
            active=True,
            awaiting_user=True,
            slots=current.slots,
            notes=current.notes,
        )


class InterviewOrchestrator:
    def __init__(self, llm_client: LlmClient, debug_mode: bool = False) -> None:
        self.llm_client = llm_client
        self.default_prompt = ""
        self.default_mode = "auto"
        self.answer_mode: str = "grounded_senior"
        self.conversation_history: list[dict] = []
        self.resume_context: str = ""
        self.resume_filename: str = ""
        self.resume_company_name: str = ""
        self.interview_context: str = DEFAULT_INTERVIEW_CONTEXT
        self.answer_style_prompt: str = CONTEXT_PROFILES[DEFAULT_INTERVIEW_CONTEXT]["style_prompt"]
        self._context_log_emitted = False
        self.debug_stream = os.getenv("ICC_DEBUG_STREAM", "").strip() == "1"
        self.debug_mode = debug_mode or os.getenv("DEBUG", "").strip().lower() == "true"
        self._request_started_at: float | None = None
        self._session_request_id = 0
        self._session_log_writer = SessionLogWriter()
        self.current_question_type = OTHER_TYPE
        self.root_question_type: str = OTHER_TYPE
        self.question_depth: str = "basic"
        self.phase_state: PhaseState | None = None
        self.current_story: dict | None = None
        self.depth_state: int = 0  # depth for current DEEP_DIVE session
        self.topic_keywords: set = set()

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
        self.current_story = None
        self.depth_state = 0
        self.topic_keywords = set()
        self.root_question_type = OTHER_TYPE

    def log_session_warning(self, message: str) -> None:
        self._session_log_writer.append_block(
            f"[WARNING | TIMESTAMP={datetime.now().isoformat(timespec='seconds')}]\n"
            f"{message}\n"
            "---\n"
        )

    def shutdown(self) -> None:
        self._session_log_writer.close(total_requests=self._session_request_id)

    def load_resume(self, path: str) -> None:
        from icc.core.resume_loader import load_resume_profile

        profile = load_resume_profile(path)
        self.resume_context = profile.formatted_resume
        self.resume_filename = profile.filename
        self.resume_company_name = profile.company_name
        self.interview_context = profile.interview_context
        self.answer_style_prompt = CONTEXT_PROFILES.get(
            self.interview_context,
            CONTEXT_PROFILES[DEFAULT_INTERVIEW_CONTEXT],
        )["style_prompt"]
        self._context_log_emitted = False

    def _emit_context_log_once(self) -> None:
        if self._context_log_emitted or not self.resume_filename:
            return
        style_preview = self.answer_style_prompt[:50]
        print(
            "[CONTEXT] Loaded resume: "
            f"{self.resume_filename} | context: {self.interview_context} | "
            f"style: {style_preview}..."
        )
        self._context_log_emitted = True

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
        resume_company = self.resume_company_name or "Unknown"
        block = (
            f"RESUME_COMPANY: {resume_company}\n"
            f"[REQUEST_ID={request_id} | TYPE={question_type} | "
            f"DEPTH={self.question_depth} | "
            f"TIMESTAMP={datetime.now().isoformat(timespec='seconds')}]\n"
            f"TRANSCRIPT: {transcript}\n"
            f"RESPONSE: {self._strip_type_prefix(response_text)}\n"
            "---\n"
        )
        self._session_log_writer.append_block(block)

    def detect_override(self, prompt: str, phase_state: PhaseState | None) -> dict:
        if re.match(
            r"^(skip|\u8df3\u8fc7|\u4e0d\u7528\u4e86|\u76f4\u63a5|next phase|\u4e0b\u4e00\u9636\u6bb5)",
            prompt.strip(),
            re.IGNORECASE,
        ):
            result = {
                "source": "user",
                "override_type": "skip",
                "target_phase": None,
            }
            logger.info("detect_override hit: source=%s type=%s", result["source"], result["override_type"])
            return result

        if not phase_state or not phase_state.active:
            return {
                "source": "none",
                "override_type": "none",
                "target_phase": None,
            }

        phase_keywords = {
            "SYSTEM_DESIGN": {
                "just design": "HLD",
                "wrap up": "WRAP",
                "architecture": "HLD",
                "design": "HLD",
                "scale": "SCALE_RELIABILITY",
                "focus on": "DEEP_DIVE",
                "talk about": "DEEP_DIVE",
            },
            "DEBUGGING_SCENARIO": {
                "root cause": "INVESTIGATE",
                "postmortem": "POSTMORTEM",
                "fix": "FIX",
                "focus on": "INVESTIGATE",
                "talk about": "INVESTIGATE",
            },
            "DEEP_DIVE": {
                "what did you learn": "IMPACT_WRAP",
                "results": "IMPACT_WRAP",
                "lesson": "IMPACT_WRAP",
                "focus on": "EXECUTION",
                "talk about": "EXECUTION",
            },
        }
        sequence = PHASE_SEQUENCES.get(phase_state.question_type, [])
        if not sequence or phase_state.phase not in sequence:
            return {
                "source": "none",
                "override_type": "none",
                "target_phase": None,
            }

        current_index = sequence.index(phase_state.phase)
        normalized_prompt = prompt.lower()
        keyword_map = phase_keywords.get(phase_state.question_type, {})
        for keyword in sorted(keyword_map, key=len, reverse=True):
            target_phase = keyword_map[keyword]
            if target_phase not in sequence:
                continue
            if not re.search(rf"(?<!\w){re.escape(keyword)}(?![\w=])", normalized_prompt):
                continue
            if sequence.index(target_phase) <= current_index:
                continue
            result = {
                "source": "interviewer",
                "override_type": "phase_jump",
                "target_phase": target_phase,
            }
            logger.info("detect_override hit: source=%s type=%s", result["source"], result["override_type"])
            return result

        return {
            "source": "none",
            "override_type": "none",
            "target_phase": None,
        }

    def _log_adaptive_event(
        self,
        source: str,
        override_type: str,
        from_phase: str | None,
        to_phase: str | None,
        skip_phase_rules: bool,
        prompt: str,
    ) -> None:
        self._debug_log(
            "adaptive_layer",
            (
                f"source={source} type={override_type} from={from_phase} "
                f"to={to_phase} skip={skip_phase_rules} prompt={prompt[:40]!r}"
            ),
        )
        self._session_log_writer.append_block(
            f"[ADAPTIVE | source={source} | override={override_type} | "
            f"from={from_phase} | to={to_phase} | "
            f"skip_phase_rules={skip_phase_rules} | "
            f"TIMESTAMP={datetime.now().isoformat(timespec='seconds')}]\n"
            f"TRIGGER: {prompt[:80]}\n"
            "---\n"
        )

    def _build_phase_override_block(self, phase_state: PhaseState) -> str:
        phase_blocks = {
            "SYSTEM_DESIGN": {
                "CLARIFY": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: CLARIFY (Requirements Alignment)\n"
                    "CRITICAL CONSTRAINT — THIS OVERRIDES ALL OTHER INSTRUCTIONS:\n"
                    "You are in the information gathering phase.\n"
                    "ANY mention of trade-offs, failure modes, alternative decisions,\n"
                    "or system components is a VIOLATION of this phase's contract.\n"
                    'Do not rationalize including them. Do not include them "briefly".\n'
                    'If you find yourself writing "I trade" or "This breaks" or\n'
                    '"I considered", STOP and delete that sentence.\n'
                    "The ONLY permitted output is: questions and default assumptions.\n"
                    "Your ONLY job this turn is to ask clarifying questions and state default assumptions.\n"
                    "Do exactly three things:\n"
                    "1. Ask at most 5 clarifying questions covering: core feature scope, scale/QPS,\n"
                    "   latency and availability SLA, data consistency, compliance or multi-tenancy constraints.\n"
                    "2. State a set of default assumptions explicitly:\n"
                    '   "If you don\'t answer, I will assume: ..."\n'
                    '3. The last sentence MUST be: "I\'ll pause here."\n'
                    "Do NOT propose any design. Do NOT mention components, databases, or architecture.\n\n"
                    "LENGTH: At most 5 questions. One sentence per question.\n"
                    "Do not explain why you are asking.\n"
                    "Do not repeat the question in different words.\n"
                    "State default assumptions in 2-3 sentences max.\n"
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Total response must be under 100 words."
                ),
                "HLD": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: HLD (High-Level Design)\n"
                    "CRITICAL CONSTRAINT — THIS OVERRIDES ALL OTHER INSTRUCTIONS:\n"
                    "You are in the high-level overview phase.\n"
                    "Do NOT write trade-offs, failure modes, or alternative decisions.\n"
                    "Do NOT go into implementation details.\n"
                    'If you find yourself writing "I trade" or "This breaks" or\n'
                    '"I considered", STOP and delete that sentence.\n'
                    "Save these for the DEEP_DIVE phase.\n"
                    "The ONLY permitted output is: component overview and data flow.\n"
                    "Give a high-level architecture overview in two paragraphs only:\n"
                    "Paragraph 1: core components and data flow (write path and read path).\n"
                    "Paragraph 2: key interface boundaries between components.\n"
                    "Do NOT go into implementation details, schema, or specific algorithms.\n"
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Then stop.\n\n"
                    "LENGTH: Two short paragraphs only. First paragraph: 3-4 sentences\n"
                    "covering components and data flow. Second paragraph: 2-3 sentences\n"
                    "on interface boundaries. Do not go beyond two paragraphs.\n"
                    "End question must be one sentence only.\n"
                    "Total response must be under 120 words."
                ),
                "DEEP_DIVE": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: DEEP_DIVE (Component Deep Dive)\n"
                    "CRITICAL CONSTRAINT — THIS OVERRIDES ALL OTHER INSTRUCTIONS:\n"
                    "You have all requirements. Do NOT ask any questions.\n"
                    "Do NOT ask about scale, QPS, payload size, latency, or constraints.\n"
                    'If you find yourself writing "can you", "what is the expected",\n'
                    '"could you provide", STOP and delete that sentence.\n'
                    'State assumptions inline: "Assuming X, I would..."\n'
                    "The ONLY permitted questions format is a single verification\n"
                    'at the very end, phrased as a statement with a question mark:\n'
                    '"What\'s your P99 target for the cache layer?"\n'
                    "Focus only on the component the interviewer selected.\n"
                    "You MUST naturally include all three of the following in your answer:\n"
                    '- Trade-off: use the phrase "I trade X for Y because..."\n'
                    '- Failure mode: use the phrase "This breaks when..."\n'
                    '- Alternative rejected: use the phrase "I considered X but rejected it because..."\n'
                    "Do NOT ask clarifying or scoping questions.\n"
                    "Focus on implementation depth, not requirements.\n"
                    'If you need to make an assumption, state it inline:\n'
                    '"Assuming X, I would..."\n'
                    "End with one specific follow-up data request to the interviewer\n"
                    "(e.g. a metric, a capacity number, a consistency requirement), then stop.\n"
                    'The last sentence MUST be: "I\'ll pause here."\n\n'
                    "LENGTH: 4-6 sentences total. Trade-off, failure mode, and alternative\n"
                    "rejected must each fit in one sentence. No sub-points or elaboration\n"
                    "unless the interviewer asks. End with one short follow-up request.\n"
                    "Total response must be under 100 words."
                ),
                "SCALE_RELIABILITY": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: SCALE_RELIABILITY (Scaling and Reliability)\n"
                    "Cover in order: capacity estimation approach, horizontal scaling strategy,\n"
                    "hotspot and sharding, caching strategy, degradation and rate limiting,\n"
                    "data recovery, monitoring and alerting signals.\n"
                    "Name at least 2 failure modes and their mitigations.\n"
                    "For each mitigation state its cost or trade-off explicitly.\n"
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Then stop.\n\n"
                    "LENGTH: Cover each topic in one sentence. Do not elaborate unless asked.\n"
                    "Name failure modes in one sentence each.\n"
                    "Total response must be under 150 words."
                ),
                "WRAP": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: WRAP (Final Summary)\n"
                    "Summarize in natural spoken English:\n"
                    "- Key architectural decisions made\n"
                    "- Two trade-offs\n"
                    "- Two failure modes\n"
                    "- At least one rejected alternative with reason\n"
                    "- One or two open questions that would need answers before production\n"
                    "Do NOT ask any new questions. This is the final output. Stop after the summary.\n\n"
                    "LENGTH: One sentence per item. Two trade-offs, two failure modes,\n"
                    "one rejected alternative, one or two open questions.\n"
                    "No elaboration. Total response must be under 120 words."
                ),
            },
            "DEBUGGING_SCENARIO": {
                "TRIAGE": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: TRIAGE (Symptom Collection)\n"
                    "CRITICAL CONSTRAINT — THIS OVERRIDES ALL OTHER INSTRUCTIONS:\n"
                    "You are in the information gathering phase.\n"
                    "ANY mention of trade-offs, failure modes, alternative decisions,\n"
                    "or system components is a VIOLATION of this phase's contract.\n"
                    'Do not rationalize including them. Do not include them "briefly".\n'
                    'If you find yourself writing "I trade" or "This breaks" or\n'
                    '"I considered", STOP and delete that sentence.\n'
                    "The ONLY permitted output is: questions and default assumptions.\n"
                    "Your ONLY job is to collect symptoms and scope the incident.\n"
                    "Ask about: user impact, error rate or latency change, start time,\n"
                    "recent deployments or config changes, affected services or regions,\n"
                    "available signals (logs, metrics, traces).\n"
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Do NOT hypothesize root cause yet.\n\n"
                    "LENGTH: Ask each question in one sentence only.\n"
                    "State the slot template in one block, not as separate paragraphs.\n"
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Total response must be under 80 words."
                ),
                "MITIGATE": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: MITIGATE (Immediate Mitigation)\n"
                    "Propose 2-3 mitigation actions ordered from lowest to highest risk.\n"
                    "For each action state: what it does, its trade-off, and possible side effects.\n"
                    'Use the phrase "I trade X for Y because..." for at least one action.\n'
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Then stop.\n\n"
                    "LENGTH: 2-3 mitigation actions. One sentence per action plus\n"
                    "one sentence for its trade-off. End question is one sentence.\n"
                    "Total response must be under 100 words."
                ),
                "INVESTIGATE": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: INVESTIGATE (Root Cause Investigation)\n"
                    "Build a hypothesis tree covering: recent deployment, dependency failure,\n"
                    "resource exhaustion, data or cache anomaly, hotspot, slow query,\n"
                    "deadlock or thread pool exhaustion.\n"
                    "For each high-priority hypothesis give concrete verification steps:\n"
                    "which specific metrics, log dimensions, or trace fields to check.\n"
                    "End with exactly one next evidence request, then stop.\n"
                    'The last sentence MUST be: "I\'ll pause here."\n\n'
                    "LENGTH: Name 3-4 hypotheses, one sentence each.\n"
                    "Verification steps: one sentence per hypothesis.\n"
                    "End with one sentence evidence request only.\n"
                    "Total response must be under 120 words."
                ),
                "FIX": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: FIX (Fix and Validation)\n"
                    "Provide a minimal hotfix and a longer-term fix.\n"
                    "For each: describe the change, rollout strategy, rollback plan,\n"
                    "and validation criteria (which signal confirms it worked).\n"
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Then stop.\n\n"
                    "LENGTH: Hotfix in 2 sentences. Long-term fix in 2 sentences.\n"
                    "Validation criteria in 1 sentence. End question in 1 sentence.\n"
                    "Total response must be under 100 words."
                ),
                "POSTMORTEM": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: POSTMORTEM (Post-Incident Review)\n"
                    "Produce a spoken postmortem covering: impact, timeline, mitigation actions,\n"
                    "root cause, action items (monitoring, testing, runbook, capacity).\n"
                    "Frame it as a learning opportunity, not blame.\n"
                    "This is the final output. Do not ask further questions. Stop after the summary.\n\n"
                    "LENGTH: Impact 1 sentence. Timeline 2 sentences. Root cause 1 sentence.\n"
                    "Action items 2-3 sentences. Total response must be under 120 words."
                ),
            },
            "DEEP_DIVE": {
                "PICK_CONTEXT": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: PICK_CONTEXT (Project Selection)\n"
                    "CRITICAL CONSTRAINT — THIS OVERRIDES ALL OTHER INSTRUCTIONS:\n"
                    "You are in the information gathering phase.\n"
                    "ANY mention of trade-offs, failure modes, alternative decisions,\n"
                    "or system components is a VIOLATION of this phase's contract.\n"
                    'Do not rationalize including them. Do not include them "briefly".\n'
                    'If you find yourself writing "I trade" or "This breaks" or\n'
                    '"I considered", STOP and delete that sentence.\n'
                    "The ONLY permitted output is: questions and default assumptions.\n"
                    "Help the candidate select and frame a project.\n"
                    "Ask about: project goal, their specific role and ownership boundary,\n"
                    "scale, and key constraints.\n"
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Do NOT discuss architecture yet.\n\n"
                    "LENGTH: Ask each question in one sentence only.\n"
                    "State the slot template in one block.\n"
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Total response must be under 80 words."
                ),
                "ARCH_OVERVIEW": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: ARCH_OVERVIEW (Architecture Overview)\n"
                    "CRITICAL CONSTRAINT — THIS OVERRIDES ALL OTHER INSTRUCTIONS:\n"
                    "You are in the high-level overview phase.\n"
                    "Do NOT write trade-offs, failure modes, or alternative decisions.\n"
                    "Do NOT go into implementation details.\n"
                    'If you find yourself writing "I trade" or "This breaks" or\n'
                    '"I considered", STOP and delete that sentence.\n'
                    "Save these for the DEEP_DIVE phase.\n"
                    "The ONLY permitted output is: component overview and data flow.\n"
                    "Describe the system boundary, key components, data flow, and critical\n"
                    "dependencies in spoken English. Do not go into implementation details.\n"
                    'The last sentence MUST be: "I\'ll pause here."\n'
                    "Then stop.\n\n"
                    "LENGTH: 3-4 sentences for the overview. End question is one sentence.\n"
                    "Total response must be under 80 words."
                ),
                "DECISIONS": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: DECISIONS (Key Decisions and Trade-offs)\n"
                    "CRITICAL CONSTRAINT — THIS OVERRIDES ALL OTHER INSTRUCTIONS:\n"
                    "You have all requirements. Do NOT ask any questions.\n"
                    "Do NOT ask about scale, QPS, payload size, latency, or constraints.\n"
                    'If you find yourself writing "can you", "what is the expected",\n'
                    '"could you provide", STOP and delete that sentence.\n'
                    'State assumptions inline: "Assuming X, I would..."\n'
                    "The ONLY permitted questions format is a single verification\n"
                    'at the very end, phrased as a statement with a question mark:\n'
                    '"What\'s your P99 target for the cache layer?"\n'
                    "Walk through the most important architectural decisions.\n"
                    "You MUST naturally include:\n"
                    '- Trade-off: "I trade X for Y because..."\n'
                    '- Failure mode: "This breaks when..."\n'
                    '- Alternative rejected: "I considered X but rejected it because..."\n'
                    "Tie each to a real constraint (SLA, compliance, cost, observability).\n"
                    "Do not add a pause marker at the end.\n"
                    "Then stop.\n\n"
                    "LENGTH: Trade-off, failure mode, and alternative rejected:\n"
                    "one sentence each. One constraint per decision.\n"
                    "End question is one sentence. Total response must be under 100 words."
                ),
                "EXECUTION": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: EXECUTION (Implementation and Operations)\n"
                    "CRITICAL CONSTRAINT — THIS OVERRIDES ALL OTHER INSTRUCTIONS:\n"
                    "You have all requirements. Do NOT ask any questions.\n"
                    "Do NOT ask about scale, QPS, payload size, latency, or constraints.\n"
                    'If you find yourself writing "can you", "what is the expected",\n'
                    '"could you provide", STOP and delete that sentence.\n'
                    'State assumptions inline: "Assuming X, I would..."\n'
                    "The ONLY permitted questions format is a single verification\n"
                    'at the very end, phrased as a statement with a question mark:\n'
                    '"What\'s your P99 target for the cache layer?"\n'
                    "Cover: the hardest engineering challenge, how correctness was validated,\n"
                    "rollout strategy, rollback plan, observability signals used,\n"
                    "and one real or hypothetical incident and how it was handled.\n"
                    "Do not add a pause marker at the end.\n"
                    "Then stop.\n\n"
                    "LENGTH: One sentence per topic. Do not elaborate unless asked.\n"
                    "End question is one sentence. Total response must be under 100 words."
                ),
                "IMPACT_WRAP": (
                    "STYLE: Natural spoken English only. No markdown, no bullet points,\n"
                    "no numbered lists, no headers. One sentence per thought.\n\n"
                    "PHASE: IMPACT_WRAP (Results and Reflection)\n"
                    "Give one quantifiable result, one trade-off or cost accepted,\n"
                    "and one thing you would improve next time.\n"
                    "This is the final output. Do not ask further questions. Stop after the summary.\n\n"
                    "LENGTH: One sentence for result. One sentence for trade-off accepted.\n"
                    "One sentence for improvement. Total response must be under 60 words."
                ),
            },
        }
        return phase_blocks.get(phase_state.question_type, {}).get(phase_state.phase, "")

    def _detect_advance(self, prompt: str, phase_state: PhaseState) -> bool:
        # Guard: FSM must be active
        if not phase_state.active:
            return False

        # Guard: terminal phase — no further advance possible
        if phase_state.phase in TERMINAL_PHASES:
            return False

        # Guard: question mark means user is asking, not confirming
        if "?" in prompt:
            return False

        text = prompt.strip().lower()

        # Only pure confirmation words trigger advance
        if re.fullmatch(
            r"(ok|okay|yes|sure|continue|proceed|next|next phase|"
            r"go ahead|got it|sounds good|好|继续|下一步|可以|行|"
            r"明白了|开始|进入下一)",
            text,
        ):
            return True

        return False

    def _is_behavioral(self, q: str) -> bool:
        """Two-layer behavioral detection: prefix patterns + verb signals."""
        q = q.lower()

        # explicit refusal/resistance signals → behavioral (highest priority)
        if any(k in q for k in ["refuse", "refused", "resist", "resisted",
                                  "pushback", "pushed back"]):
            return True

        # Exclude clear DEEP_DIVE / SYSTEM_DESIGN signals first
        if any(k in q for k in [
            "design a system",
            "how would you design",
            "describe how you designed",
            "describe how you built",
            "describe how you architected",
            "how does it work",
        ]):
            return False

        # Layer 1: strong prefix patterns
        if any(p in q for p in BEHAVIORAL_PATTERNS):
            return True

        # Layer 2: verb-anchored phrases + behavioral verb
        if any(k in q for k in [
            "describe how you",
            "how did you",
        ]) and any(v in q for v in BEHAVIORAL_VERBS):
            return True

        # Layer 3: hardest/most difficult + problem noun
        if any(k in q for k in [
            "hardest", "most difficult", "toughest", "worst"
        ]) and any(v in q for v in [
            "bug", "issue", "problem", "decision", "mistake"
        ]):
            return True

        return False

    def _classify_question(self, transcript: str, has_image: bool) -> str:
        if "[This is a follow-up" in transcript:
            self.question_depth = "basic"
            return "FOLLOW_UP"

        if has_image:
            self.question_depth = "deep"
            return "CODING"

        q = transcript.lower()

        # BEHAVIORAL override — two-layer pattern+verb detection
        if self._is_behavioral(q):
            self.question_depth = "practical"
            return "BEHAVIORAL"

        groq_api_key = self.llm_client.config.groq_api_key
        if not groq_api_key:
            logger.warning("GROQ_API_KEY missing, skipping Groq classification and using OTHER")
            self.question_depth = "basic"
            return OTHER_TYPE

        started_at = perf_counter()
        try:
            raw_result = self.llm_client.complete_text(
                system=(
                    "Classify the interview question into TYPE and DEPTH.\n\n"
                    "TYPE (choose exactly one):\n"
                    "- BEHAVIORAL: interpersonal situations, conflict, teamwork, soft skills,\n"
                    "              how you handled people or process challenges;\n"
                    "              also: past technical achievements framed as 'Tell me about\n"
                    "              a time you...' — classify as BEHAVIORAL + practical\n"
                    "- CODING: algorithm, data structure, or implementation problem.\n"
                    "  Key signals: 'given an array/tree/list/graph', 'implement a function',\n"
                    "  'return the result', 'find/detect/count/reverse/sort something'.\n"
                    "  These are LeetCode-style problems, NOT projects you built.\n"
                    "- CONCEPTUAL: define a concept, explain how something works,\n"
                    "              internal mechanisms, or production usage\n"
                    "- SYSTEM_DESIGN: design a large-scale distributed system end to end\n"
                    "- DEEP_DIVE: walk me through a system you built, an architectural decision\n"
                    "             you made, or a technical project you owned end to end;\n"
                    "             also: 'Describe how you designed...', 'how did you handle X'\n"
                    "             when asked about past work (past-tense ownership pattern)\n"
                    "- DEBUGGING_SCENARIO: a production problem requiring diagnosis and\n"
                    "  decision-making — memory leaks, CPU spikes, error rates, outages,\n"
                    "  latency issues, or any hypothetical situation requiring immediate action\n"
                    "- FOLLOW_UP: asking to elaborate, clarify, or continue a previous answer\n\n"
                    "DEPTH (choose exactly one):\n"
                    "- basic      (definition or simple explanation)\n"
                    "- practical  (how it works in real systems)\n"
                    "- deep       (full system design or architecture)\n\n"
                    "Rules:\n"
                    "- 'What is X' → CONCEPTUAL + basic\n"
                    "- 'How do you handle / implement X' → CONCEPTUAL + practical\n"
                    "- 'Design a system that does X end-to-end' → SYSTEM_DESIGN + deep\n"
                    "- 'how you designed...' / 'Describe how you designed...' / past-tense ownership → DEEP_DIVE + practical\n\n"
                    "DEPTH rules:\n"
                    "- 'What is X' or 'Define X' → basic\n"
                    "- 'How do you handle X' or 'How do you implement X' → practical\n"
                    "- 'Design a system that does X end-to-end' → deep\n"
                    "- 'Tell me about a system you built' → practical\n"
                    "- 'Tell me about a time you...' → practical (always, regardless of topic)\n"
                    "- 'Walk me through...' → practical\n"
                    "When in doubt between basic and practical, choose practical.\n\n"
                    "Output format: [TYPE:xxx] [DEPTH:yyy]\n"
                    "Use exact TYPE names from the list above."
                ),
                prompt=(
                    f"Question: {transcript}\n"
                    f"Has image: {has_image}\n"
                    "Classification:"
                ),
                model=CLASSIFIER_MODEL,
                max_tokens=40,
                api_key=groq_api_key,
                base_url=self.llm_client.config.groq_base_url,
                timeout=CLASSIFIER_TIMEOUT_SECONDS,
            )
            if self.debug_mode:
                print(f"[CLASSIFIER RAW] {raw_result!r}")

            question_type, parsed_depth = parse_classifier_output(raw_result)
            self.question_depth = parsed_depth

            elapsed_ms = (perf_counter() - started_at) * 1000
            logger.info(
                "Classified as %s depth=%s (%.0f ms)", question_type, parsed_depth, elapsed_ms
            )
            return question_type
        except RuntimeError as exc:
            logger.warning("Groq classify failed, using OTHER: %s", exc)
            self.question_depth = "basic"
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
        depth: int = 0,
    ) -> str:
        addon = ""
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
        example_phrase = random.choice(TRADEOFF_STYLES)
        grounded_senior_addon = (
            "GROUNDED SENIOR MODE ACTIVE.\n\n"
            "Your answer MUST cover ALL of the following — no exceptions:\n"
            "- Why you chose one approach over another\n"
            "- One limitation or failure scenario specific to this design\n"
            "- A briefly mentioned alternative, if it helps clarity\n\n"
            "Do NOT use fixed phrases repeatedly. Vary sentence structure naturally.\n"
            f"Example of preferred phrasing: \"{example_phrase}\"\n\n"
            "Do NOT invent exotic failure scenarios "
            "(e.g., rare distributed partition edge cases)\n"
            "unless clearly framed as hypothetical.\n"
            "Prefer failures common in backend systems "
            "tied to real observability signals "
            "(Prometheus metrics, GC logs, slow query logs).\n\n"
            "For BEHAVIORAL: name the specific project from "
            "resume in sentence 1. Include one measurable result.\n"
            "State one concrete trade-off: 'I chose X over Y because [specific reason].'\n"
            "BANNED: 'worked reliably', 'ensured scalability', 'improved performance'\n\n"
            "Integrate these elements into your explanation. "
            "Do not list them as a separate section at the end.\n"
        )
        if question_type in {
            "SYSTEM_DESIGN",
            "DEBUGGING_SCENARIO",
            "DEEP_DIVE",
            "CONCEPTUAL_DEEP",
            "BEHAVIORAL",
        }:
            if answer_mode == "grounded_senior" and self.interview_context != "enterprise":
                addon = grounded_senior_addon
            elif answer_mode in {"grounded_senior", "senior"}:
                addon = senior_addon
        elif question_type == "CODING":
            addon = senior_addon
        # Build the conceptual depth instruction based on current session depth
        if question_type == "CONCEPTUAL_DEEP":
            conceptual_instruction = (
                "Start with a clear explanation of the concept in 1-2 sentences,\n"
                "then expand into trade-offs or implementation details.\n"
                "Ground your answer in real-world systems where applicable,\n"
                "but do NOT force personal experience if not directly relevant.\n"
                "\n"
                "You MUST include ALL of the following:\n"
                "\n"
                "1. TRADE-OFF (REQUIRED)\n"
                "   'I chose X over Y because [specific technical reason]'\n"
                "   X and Y must be concrete alternatives, not vague.\n"
                "\n"
                "2. FAILURE MODE (REQUIRED if relevant to the question)\n"
                "   If the concept involves failure scenarios (e.g., retries, consistency,\n"
                "   distributed state), you MUST include one specific failure.\n"
                "   If the concept is purely mechanical (e.g., thread model, GC algorithm),\n"
                "   skip this or mention briefly only if it adds clarity.\n"
                "   Do NOT force a failure mode that does not naturally apply.\n"
                "\n"
                "3. MECHANISM (REQUIRED)\n"
                "   Explain how it actually works in practice.\n"
                "   Use concrete phrases where applicable:\n"
                "   'State is stored in...' / 'We persist...' /\n"
                "   'Recovery works by...' / 'We ensure idempotency by...'\n"
                "   Do NOT force storage or persistence if the concept is purely computational.\n"
                "\n"
                "If TRADE-OFF or MECHANISM is missing -> REWRITE before returning.\n"
                "FAILURE MODE missing is only a violation if the concept involves failure.\n"
                "\n"
                "BANNED phrases: 'worked reliably', 'ensured scalability',\n"
                "'improved performance', 'robust system'\n"
                "\n"
                "Length: 300 tokens max.\n"
            )
        elif (
            question_type == "CONCEPTUAL_BASIC"
            and self.question_depth == "practical"
        ):
            conceptual_instruction = (
                "CONCEPTUAL (PRACTICAL): "
                "Explain how this is handled in real systems. "
                "Start with a direct answer in one sentence. "
                "Explain practical implementation approaches. "
                "Mention common patterns used in production. "
                "Include a brief trade-off if relevant. "
                "Do NOT produce a full system design. "
                "Keep it concise and grounded in real experience. "
                "Stay within 300 tokens.\n"
            )
        elif question_type == "CONCEPTUAL_BASIC":
            conceptual_instruction = (
                "CONCEPTUAL_BASIC: stay within 120 tokens.\n"
                "Structure your answer in exactly this order:\n"
                "1. DEFINITION: one clear sentence. No jargon unless you immediately explain it.\n"
                "2. WHY IT MATTERS: one real-world consequence if ignored.\n"
                "3. OPTIONAL EXAMPLE: a simple example.\n"
                "   Prefer intuitive examples over production stories.\n"
                "   Use your own system only if it clearly improves clarity.\n"
                "Do NOT expand into tradeoffs, failure scenarios, or deep architecture.\n"
                "Do NOT over-explain. Basic questions reward clarity, not volume.\n"
                "Avoid turning this into a system design answer.\n"
            )
        else:
            conceptual_instruction = "CONCEPTUAL_DEEP: stay within 300 tokens.\n"

        if question_type == "BEHAVIORAL" and self.question_depth == "practical":
            behavioral_instruction = (
                "BEHAVIORAL (PRACTICAL): Name the specific project and your role "
                "in sentence 1. State the problem or goal in one sentence.\n\n"
                "Use the following rules to determine intent:\n"
                "- If the question includes words like:\n"
                "  'disagree', 'conflict', 'influence', 'convince', 'push back',\n"
                "  'stakeholder', 'negotiate'\n"
                "  → use FRICTION ARC\n"
                "- If the question includes words like:\n"
                "  'mistake', 'bug', 'failed', 'slow', 'performance', 'debug'\n"
                "  → use TECHNICAL BEHAVIORAL\n"
                "- If unclear, default to TECHNICAL BEHAVIORAL.\n\n"
                "FRICTION ARC (for conflict/negotiation/influence questions):\n"
                "- Focus on people, decisions, and communication\n"
                "- Only include technical details if they directly support the conflict\n"
                "- Do NOT turn this into a technical deep-dive\n"
                "- If more than 50% of your answer is technical explanation, REWRITE\n"
                "  to focus on decision and communication.\n"
                "You MUST include ALL of the following:\n"
                "1. FRICTION: explicitly state what the other party wanted and\n"
                "   WHY they believed it was correct — based on their specific\n"
                "   experience, constraints, or role (not generic concerns).\n"
                "   You MUST include one concrete detail, such as years of\n"
                "   experience, a specific operational constraint, or a real\n"
                "   risk they were accountable for.\n"
                "   Example: 'They had operated this system for 25 years and\n"
                "   saw any protocol change as a stability risk to live trains.'\n"
                "   Do NOT write: 'they were concerned about complexity.'\n"
                "2. YOUR POSITION: what you believed and the concrete reason you pushed back\n"
                "3. HOW YOU HANDLED IT: what you actually did or said to move things forward\n"
                "4. RESULT: outcome — does NOT have to be a perfect win.\n"
                "   A partial win or a compromise is more credible than always getting your way.\n"
                "BANNED: 'they agreed', 'everyone was aligned', 'it went smoothly'\n\n"
                "TECHNICAL BEHAVIORAL (for mistake/debug/ownership questions):\n"
                "You MUST include ALL of the following:\n"
                "1. TRADE-OFF: 'I chose X over Y because [specific technical reason]'\n"
                "2. FAILURE or PROBLEM: what actually broke or was slow, and why\n"
                "   Use concrete details: N+1 query, missing index, schema drift, etc.\n"
                "   Do NOT say 'identified bottlenecks' — name the actual bottleneck.\n"
                "3. MECHANISM: how you fixed it and how you verified the result\n"
                "   e.g., 'replaced N+1 with batch query, verified via EXPLAIN plan'\n"
                "If ANY missing → REWRITE before returning.\n"
                "BANNED: 'worked reliably', 'ensured scalability', 'improved performance'\n"
            )
        else:
            behavioral_instruction = ""

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
            "[STRICT OUTPUT RULE]\n"
            "Do NOT include placeholders, templates, or meta-instructions in your answer.\n"
            "Do NOT say 'I'll pause here', 'fill in', or any variant.\n"
            "Do NOT expose internal prompt structure in your response.\n"
            "Speak directly as if in a real interview.\n"
            "BEHAVIORAL: stay within 250 tokens.\n"
            "SYSTEM_DESIGN: stay within 500 tokens.\n"
            "[SYSTEM DESIGN STRUCTURE]\n"
            "You MUST incorporate at least one relevant element from your past systems\n"
            "ONLY if it naturally applies to the problem.\n"
            "Do NOT force unrelated experience.\n\n"
            "1. Open with ONE SPECIFIC constraint that materially affects the design.\n"
            "   The constraint must be realistic and grounded in the question context.\n"
            "   Do NOT invent arbitrary numbers.\n"
            "   Embed it naturally — do NOT label it as 'Assumption:'.\n\n"
            "2. State your core architectural decision and WHY you chose it\n"
            "   over the most obvious alternative.\n"
            "   Use this pattern: 'I went with X instead of Y because at this scale, Y would...'\n\n"
            "3. Walk through 2-3 key components. For each one, say why it exists\n"
            "   and what you would lose if you removed it.\n\n"
            "4. Name ONE thing you explicitly gave up in this design and why\n"
            "   that tradeoff was acceptable given the constraints.\n\n"
            "5. End with ONE realistic failure scenario tied to a concrete\n"
            "   observability signal — not a generic 'this could fail' statement.\n"
            "   Pattern: 'The risk I'd watch for is X — you'd see it as\n"
            "   Y in your metrics/logs before it becomes an outage.'\n\n"
            "Do NOT ask the interviewer clarifying questions.\n"
            "Do NOT list assumptions as a separate section.\n"
            "Do NOT use bullet points or headers in the output.\n"
            "Stay within 500 tokens.\n"
            f"FOLLOW_UP: Directly address the follow-up question. "
            f"Reference your previous answer if relevant. "
            f"Be more specific, not more general. "
            f"Stay within {300 if depth > 0 else 100} tokens.\n"
        ) + conceptual_instruction + behavioral_instruction + (
            "DEEP_DIVE: Tell the story of a real system you built.\n"
            "Structure: what it was → how it started → how it evolved → result.\n"
            "Include at least one specific metric or outcome.\n"
            "Keep it conversational, 3-4 sentences minimum.\n"
            "SELF-CHECK before returning:\n"
            "- Does the answer contain 'worked reliably'? → REMOVE and replace\n"
            "  with the specific reason (e.g., 'because Kafka gave us replay\n"
            "  capability and backpressure handling')\n"
            "- Does the answer contain 'ensured scalability' or\n"
            "  'improved performance'? → REMOVE and replace with concrete metric\n"
            "  or mechanism\n"
            "- Is 'I chose X over Y because' present? If not → REWRITE\n"
            "- Is a specific failure named? If not → REWRITE\n"
            "Stay within 450 tokens.\n"
            "DEBUGGING_SCENARIO: stay within 280 tokens.\n"
            "CODING: Start with a brief approach in 1-2 sentences. "
            "Then provide executable code that directly answers the problem. "
            "After the code, briefly mention time complexity, space complexity, and one edge case. "
            "Prefer complete solutions over high-level pseudocode. "
            "Stay within 900 tokens.\n"
            "For CODING only, code blocks are allowed. "
            "For all other question types, never use markdown formatting: no bold, no bullet points, no emoji, and no numbered lists.\n"
            "For CODING, use concise explanation plus code. "
            "For all other question types, answers must read as natural spoken English only.\n"
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
        fsm_enabled: bool = False,
    ) -> None:
        self.answer_mode = answer_mode
        self._request_started_at = perf_counter()
        self._session_request_id += 1
        request_id = self._session_request_id
        worker = Thread(
            target=self._run_request,
            args=(request_id, prompt, on_chunk, on_complete, on_error, images_b64, answer_mode, fsm_enabled),
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
        fsm_enabled: bool = False,
    ) -> None:
        history = self.conversation_history
        has_image = bool(images_b64)
        _is_follow = False

        # Step 1: Classification — lock to A-class type once session starts
        if self.phase_state is not None and self.phase_state.active:
            # A-class session in progress (active or just completed):
            # never re-classify, always use the locked question_type
            question_type = self.phase_state.question_type
            logger.info(
                "Classification skipped: locked to %s "
                "(phase=%s active=%s)",
                question_type,
                self.phase_state.phase,
                self.phase_state.active,
            )
        else:
            question_type = self._classify_question(prompt, has_image)

        question_class = QUESTION_CLASS_MAP.get(question_type, "B")

        is_new_question = False  # default; updated below if phase_state is active

        # New question detection: only run if phase_state exists
        # and current input looks like a new question
        if self.phase_state is not None and self.phase_state.active:
            words = prompt.strip().split()
            slot_fill_pattern = re.compile(
                r"(scale|qps|sla|fr|latency|consistency|constraints|"
                r"scope|throughput|availability)\s*[:=]",
                re.IGNORECASE,
            )
            question_verbs = re.compile(
                r"\b(design|build|implement|create|explain|describe|"
                r"how would you|walk me through|tell me about|"
                r"talk me through|what would you|how did you|"
                r"give me an example|production alert|incident|"
                r"error rate|spiked|outage|debugging|"
                r"what is|what are|compare|difference between)\b",
                re.IGNORECASE,
            )
            is_slot_fill = bool(slot_fill_pattern.search(prompt))
            is_new_question = (
                not _is_follow and  # follow-up exemption — must be first
                not is_slot_fill
                and (
                    # Signal 1: explicit question verb/keyword or ends with ?
                    (
                        len(words) >= 4
                        and (
                            prompt.strip().endswith("?")
                            or question_verbs.search(prompt)
                        )
                    )
                    # Signal 2: long input without advance pattern
                    or len(words) >= 8
                )
                and not any(
                    re.search(p, prompt.strip(), re.IGNORECASE)
                    for p in ADVANCE_PATTERNS
                )
            )
            if is_new_question:
                # Re-classify the new question from scratch
                question_type = self._classify_question(prompt, has_image)
                question_class = QUESTION_CLASS_MAP.get(question_type, "B")
                logger.info(
                    "New question detected (was %s v=%s) → "
                    "re-classified as %s, starting new session",
                    self.phase_state.phase,
                    self.phase_state.phase_version,
                    question_type,
                )
                if question_class == "A":
                    logger.info(
                        "New question detected, resetting phase_state "
                        "(was %s v=%s) → starting new session",
                        self.phase_state.phase if self.phase_state else None,
                        self.phase_state.phase_version if self.phase_state else 0,
                    )
                    self.phase_state = PhaseState.start(question_type)
                else:
                    self.phase_state = None
                self.current_question_type = question_type
                if question_type != "FOLLOW_UP":
                    self.root_question_type = question_type
                self.conversation_history = []
                history = self.conversation_history

        # Re-classify FOLLOW_UP if it introduces a new topic
        if question_type == "FOLLOW_UP" and introduces_new_topic(prompt, self.topic_keywords):
            question_type = self._classify_question(prompt, has_image)
            question_class = QUESTION_CLASS_MAP.get(question_type, "B")
            self.depth_state = 0
            self.topic_keywords = set()
            if question_type != "FOLLOW_UP":
                self.root_question_type = question_type

        # Depth state management — always runs for DEEP_DIVE context, independent of FSM
        is_deep_dive_context = (
            question_type == "DEEP_DIVE" or
            (question_type == "FOLLOW_UP" and self.root_question_type == "DEEP_DIVE")
        )

        if is_deep_dive_context:
            _is_follow = is_followup(prompt, self.phase_state, self.topic_keywords)
            if _is_follow:
                self._debug_log("followup_detected", prompt[:60])
                self.depth_state = map_to_depth(prompt, self.depth_state)
                self.depth_state = min(self.depth_state, MAX_DEPTH)
            elif is_new_question:
                self.depth_state = 0
                self.topic_keywords = set()
            # else: advance signal or short confirmation — leave depth unchanged
        else:
            _is_follow = False

        logger.info(
            "phase_state=%s question_class=%s",
            self.phase_state,
            question_class,
        )
        override = self.detect_override(prompt, self.phase_state)
        skip_phase_rules = False
        from_phase = self.phase_state.phase if self.phase_state else None

        if override["source"] == "user":
            skip_phase_rules = True
            logger.info(
                "[OVERRIDE_APPLIED] source=%s skip_phase_rules=True "
                "from_phase=%s",
                override["source"],
                from_phase,
            )
            to_phase = None
        elif override["source"] == "interviewer":
            target = override["target_phase"]
            if self.phase_state and target is not None:
                self.phase_state = PhaseState(
                    question_type=self.phase_state.question_type,
                    phase=target,
                    phase_version=self.phase_state.phase_version + 1,
                    active=True,
                    awaiting_user=True,
                    slots=self.phase_state.slots,
                    notes=self.phase_state.notes,
                )
                logger.info(
                    "[OVERRIDE_APPLIED] source=%s target_phase=%s "
                    "from_phase=%s v=%s",
                    override["source"],
                    override["target_phase"],
                    from_phase,
                    self.phase_state.phase_version,
                )
            to_phase = target
        else:
            to_phase = None

        self._log_adaptive_event(
            source=override["source"],
            override_type=override["override_type"],
            from_phase=from_phase,
            to_phase=to_phase,
            skip_phase_rules=skip_phase_rules,
            prompt=prompt,
        )
        logger.info(
            "after_adaptive: override=%s skip_phase_rules=%s",
            override,
            skip_phase_rules,
        )

        # Terminal phase early-exit: short confirmations after WRAP/etc. are noise.
        if (
            self.phase_state is not None
            and self.phase_state.phase in TERMINAL_PHASES
            and self.phase_state.active
        ):
            words_check = prompt.strip().split()
            if len(words_check) <= 3:
                self._debug_log(
                    "terminal_phase_skip",
                    f"phase={self.phase_state.phase} prompt={prompt[:40]!r} "
                    f"len={len(words_check)} → skipping LLM call",
                )
                on_complete(" ")
                return

        logger.debug(
            "advance_check: override_source=%s skip=%s "
            "phase=%s active=%s",
            override["source"],
            skip_phase_rules,
            self.phase_state.phase if self.phase_state else None,
            self.phase_state.active if self.phase_state else None,
        )
        if (
            question_class == "A"
            and self.phase_state is not None
            and self.phase_state.active
            and not skip_phase_rules
            and override["source"] == "none"
        ):
            if self._detect_advance(prompt, self.phase_state):
                old_phase = self.phase_state.phase
                self.phase_state = PhaseState.advance(self.phase_state)
                self._debug_log(
                    "phase_advance",
                    f"phase {old_phase!r} → {self.phase_state.phase!r} "
                    f"v={self.phase_state.phase_version} "
                    f"active={self.phase_state.active}"
                )
                self._session_log_writer.append_block(
                    f"[PHASE_ADVANCE | from={old_phase} | "
                    f"to={self.phase_state.phase} | "
                    f"v={self.phase_state.phase_version} | "
                    f"TIMESTAMP={datetime.now().isoformat(timespec='seconds')}]\n"
                    f"TRIGGER: {prompt[:80]}\n"
                    "---\n"
                )
        if self.phase_state:
            logger.info(
                "after_advance: phase=%s version=%s active=%s",
                self.phase_state.phase,
                self.phase_state.phase_version,
                self.phase_state.active,
            )

        if question_class == "A" and not skip_phase_rules and fsm_enabled:
            # Reset stale ended sessions so every new Class A question starts at CLARIFY
            if self.phase_state is None or not self.phase_state.active:
                self.phase_state = PhaseState.start(question_type)
        elif not fsm_enabled and question_class == "A":
            self.phase_state = None

        self.current_question_type = question_type
        if question_type != "FOLLOW_UP":
            self.root_question_type = question_type
        question = prompt.strip() or self.default_prompt

        # --- Story bank routing ---
        if question_type in ("DEEP_DIVE", "BEHAVIORAL"):
            story = find_story(question)
            if story:
                self.current_story = story
                # Inject story as context — do NOT bypass FSM
                self._story_context = (
                    "IMPORTANT: Base your answer ONLY on this system. "
                    "Do NOT introduce a different project or company.\n\n"
                    f"{get_summary(story)}\n\n"
                    "Do NOT introduce unrelated systems or fabricated details.\n"
                    "You MAY use standard distributed systems terminology where needed "
                    "(e.g., at-least-once, exactly-once, idempotency, atomicity)."
                )
                # Extract topic keywords from story for follow-up detection
                self.topic_keywords = set(story["tags"])
                # Fall through to normal FSM pipeline — do NOT return
            # No story match → fall through to normal pipeline

        if question_type == "FOLLOW_UP":
            if self.current_story:
                snippet = None
                if is_experience_followup(question):
                    snippet = get_snippet(self.current_story, question)
                if snippet:
                    # Inject snippet as answer base — let LLM adapt angle to question
                    self._snippet_context = snippet
                    # Fall through to normal LLM pipeline
                else:
                    # Path B: LLM with story constraint
                    story_reminder = (
                        "\nIMPORTANT: Answer based ONLY on this system:\n"
                        f"{get_summary(self.current_story)}\n"
                        "HARD RULES:\n"
                        "- You MUST use at least one concrete detail from "
                        "this system in your answer\n"
                        "- Do NOT mention any other company or project\n"
                        "- Do NOT introduce details not present above\n"
                        "- If asked why X over Y, answer using only the "
                        "context of this system\n"
                        "- Stay within this project's scope for the "
                        "entire answer\n"
                    )
                    self._story_reminder = story_reminder
            # Fall through to normal FOLLOW_UP pipeline
        # --- End story bank routing ---

        system_message, _ = self.build_prompt(
            prompt=prompt,
            answer_mode=answer_mode,
            question_type=question_type,
            phase_state=self.phase_state,
            skip_phase_rules=skip_phase_rules,
            fsm_enabled=fsm_enabled,
            depth=self.depth_state,
            root_question_type=self.root_question_type,
        )
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
        if question_class == "A":
            self.conversation_history = self.conversation_history[-8:]
        else:
            self.conversation_history = self.conversation_history[-2:]
        on_complete(response_text)

    def build_prompt(
        self,
        prompt: str,
        answer_mode: str = "grounded_senior",
        question_type: str = OTHER_TYPE,
        phase_state: PhaseState | None = None,
        skip_phase_rules: bool = False,
        fsm_enabled: bool = False,
        depth: int = 0,
        root_question_type: str = OTHER_TYPE,
    ) -> tuple[str, str]:
        question_class = QUESTION_CLASS_MAP.get(question_type, "B")
        # Route by session semantics, not single-turn type.
        # FOLLOW_UP inside a DEEP_DIVE session should use DEEP_DIVE prompt.
        is_deep_dive_context = (
            question_type == "DEEP_DIVE" or
            (question_type == "FOLLOW_UP" and root_question_type == "DEEP_DIVE")
        )
        effective_question_type = "DEEP_DIVE" if is_deep_dive_context else question_type
        logger.info("build_prompt: phase_state=%s skip=%s", phase_state, skip_phase_rules)
        question = prompt.strip() or self.default_prompt
        resume = self.resume_context or "No resume provided."
        base_system_message = self._system_prompt_template(
            answer_mode=answer_mode,
            question_type=effective_question_type,
            depth=depth,
        ).format(resume=resume)
        # --- Context card injection ---
        cards = select_cards(question)
        logger.debug("[context_cards] selected: %d card(s) for question: %s", len(cards), question[:60])
        if cards:
            card_block = "## Engineering Context (from my past projects)\n"
            for card in cards:
                card_block += f"- {card}\n"
            base_system_message = card_block + "\n" + base_system_message
        # --- End context card injection ---
        base_system_message += (
            "\n\n## Answer Style Guidance\n"
            f"{self.answer_style_prompt}"
        )
        system_message = base_system_message

        if phase_state is not None and not skip_phase_rules:
            phase_block = self._build_phase_override_block(phase_state)
            if phase_block:
                system_message += "\n\n" + phase_block

        # story_reminder injection — must come before depth_block
        story_reminder = getattr(self, '_story_reminder', None)
        if story_reminder:
            system_message += story_reminder
            self._story_reminder = None  # consume after use

        # story_context injection — must come before depth_block
        story_context = getattr(self, '_story_context', None)
        if story_context:
            system_message += "\n\n" + story_context
            self._story_context = None

        snippet_context = getattr(self, '_snippet_context', None)
        if snippet_context:
            system_message += (
                "\n\nANSWER BASE — USE THIS AS YOUR STARTING POINT:\n"
                f"{snippet_context}\n\n"
                "Adapt the phrasing and angle to match the exact question asked. "
                "Keep the core facts and guarantee structure intact. "
                "Do not change the guarantee declaration format: "
                "'I guarantee X, but I do NOT guarantee Y.'"
            )
            self._snippet_context = None

        # depth_block injection — always last, highest priority
        if is_deep_dive_context:
            if phase_state is not None and phase_state.phase == "PICK_CONTEXT":
                depth_block = ""  # PICK_CONTEXT is info-gathering only, no content
            elif phase_state is not None and phase_state.phase == "ARCH_OVERVIEW":
                depth_block = build_depth_block(min(depth, 1))  # guarantee only
            else:
                depth_block = build_depth_block(depth)
            if depth_block:
                system_message += "\n\n" + depth_block
        self._emit_context_log_once()
        user_message = f"Question: {question}"
        return system_message, user_message
