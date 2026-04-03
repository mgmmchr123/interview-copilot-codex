from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from time import perf_counter

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import AppConfig


SYSTEM_DESIGN_QUESTIONS = [
    "Design a rate limiting system for a public API.",
    "Design a distributed cache like Redis.",
    "How would you design a notification system for millions of users?",
    "Design a URL shortener like bit.ly.",
    "How would you architect a payment processing system?",
]

SHOULD_NOT_BE_SYSTEM_DESIGN = [
    "How do you handle duplicate messages in a distributed system?",
    "How do you ensure data consistency across microservices?",
    "How would you implement idempotency in a REST API?",
    "How do you handle distributed transactions?",
    "What's your approach to handling race conditions in a database?",
]

MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

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
    ("SYSTEM_DESIGN", r"\bSYSTEM_DESIGN\b"),
    ("BEHAVIORAL", r"\bBEHAVIORAL\b"),
    ("CODING", r"\bCODING\b"),
    ("FOLLOW_UP", r"\bFOLLOW_UP\b"),
    ("DEEP_DIVE", r"\bDEEP_DIVE\b"),
    ("CONCEPTUAL", r"\bCONCEPTUAL\b"),
]
TYPE_PATTERN = re.compile(
    r"\[(?:T|TYPE[S]?)\s*:\s*([A-Z_]+)",
    re.IGNORECASE,
)
DEPTH_PATTERN = re.compile(
    r"\[(?:D|DEPTH)\s*:\s*(basic|practical|deep|[BPD])\b"
    r"|\b(DEEP|PRACTICAL|BASIC)\b(?!\s*_)",
    re.IGNORECASE,
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


def parse_classifier_output(raw: str) -> tuple[str, str]:
    type_match = TYPE_PATTERN.search(raw)
    depth_match = DEPTH_PATTERN.search(raw)

    if type_match:
        code = type_match.group(1).upper()
        parsed_type = TYPE_CODE_MAP.get(code) or (
            code if code in QUESTION_TYPES else None
        )
    else:
        parsed_type = _extract_type(raw)

    if depth_match:
        val = (depth_match.group(1) or depth_match.group(2)).upper()
        parsed_depth = DEPTH_CODE_MAP.get(val) or val.lower()
        if parsed_depth not in VALID_DEPTHS:
            parsed_depth = "basic"
    else:
        parsed_depth = "basic"

    if parsed_type == "CONCEPTUAL":
        question_type = (
            "CONCEPTUAL_DEEP" if parsed_depth != "basic"
            else "CONCEPTUAL_BASIC"
        )
    elif parsed_type in QUESTION_TYPES:
        question_type = parsed_type
    else:
        question_type = OTHER_TYPE

    return question_type, parsed_depth


def _call_classifier(client: OpenAI, model: str, question: str) -> tuple[str, float]:
    started_at = perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {
                "role": "user",
                "content": f"Question: {question}\nHas image: False\nClassification:",
            },
        ],
        max_tokens=40,
        timeout=1.0,
    )
    latency_ms = (perf_counter() - started_at) * 1000
    raw_output = response.choices[0].message.content or ""
    return raw_output.strip(), latency_ms


def _run_true_positives(
    client: OpenAI,
    model: str,
) -> tuple[int, list[float]]:
    print("--- TRUE POSITIVES (should be SYSTEM_DESIGN) ---")
    correct_count = 0
    latencies: list[float] = []

    for index, question in enumerate(SYSTEM_DESIGN_QUESTIONS, start=1):
        try:
            raw_output, latency_ms = _call_classifier(client, model, question)
            parsed_type, parsed_depth = parse_classifier_output(raw_output)
            is_correct = parsed_type == "SYSTEM_DESIGN"
        except Exception as exc:
            raw_output = f"<ERROR: {exc}>"
            latency_ms = 0.0
            parsed_type, parsed_depth = ("OTHER", "basic")
            is_correct = False

        if is_correct:
            correct_count += 1
        latencies.append(latency_ms)

        print(f'Q{index}: "{question}"')
        print(f"  RAW:    {raw_output!r}")
        print(f"  PARSED: type={parsed_type} depth={parsed_depth}")
        print(f"  CORRECT: {'✅' if is_correct else '❌'}  |  latency: {latency_ms:.0f}ms")
        print()

    return correct_count, latencies


def _run_true_negatives(
    client: OpenAI,
    model: str,
) -> tuple[int, int, list[float]]:
    print("--- TRUE NEGATIVES (should NOT be SYSTEM_DESIGN) ---")
    correct_count = 0
    false_positives = 0
    latencies: list[float] = []

    for index, question in enumerate(SHOULD_NOT_BE_SYSTEM_DESIGN, start=1):
        try:
            raw_output, latency_ms = _call_classifier(client, model, question)
            parsed_type, parsed_depth = parse_classifier_output(raw_output)
            is_correct = parsed_type != "SYSTEM_DESIGN"
        except Exception as exc:
            raw_output = f"<ERROR: {exc}>"
            latency_ms = 0.0
            parsed_type, parsed_depth = ("OTHER", "basic")
            is_correct = False

        if is_correct:
            correct_count += 1
        elif parsed_type == "SYSTEM_DESIGN":
            false_positives += 1
        latencies.append(latency_ms)

        print(f'Q{index}: "{question}"')
        print(f"  RAW:    {raw_output!r}")
        print(f"  PARSED: type={parsed_type} depth={parsed_depth}")
        if is_correct:
            print(f"  CORRECT: ✅  |  latency: {latency_ms:.0f}ms")
        else:
            print(
                f"  CORRECT: ❌ (false positive — got {parsed_type})  |  "
                f"latency: {latency_ms:.0f}ms"
            )
        print()

    return correct_count, false_positives, latencies


def main() -> int:
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_api_key:
        print("GROQ_API_KEY is required.")
        return 1

    config = AppConfig.from_env()
    client = OpenAI(
        api_key=groq_api_key,
        base_url=config.groq_base_url,
        timeout=1.0,
        max_retries=0,
    )

    summary_rows: list[tuple[str, int, int, int, float]] = []

    for model in MODELS:
        print(f"=== MODEL: {model} ===")
        print()

        tp_count, tp_latencies = _run_true_positives(client, model)
        tn_count, false_positives, tn_latencies = _run_true_negatives(client, model)

        all_latencies = tp_latencies + tn_latencies
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
        summary_rows.append(
            (model, tp_count, tn_count, false_positives, avg_latency)
        )

    print("=== SUMMARY ===")
    print("Model                   | TP (5) | TN (5) | False Positives | Avg Latency")
    for model, tp_count, tn_count, false_positives, avg_latency in summary_rows:
        print(
            f"{model:<23} | {tp_count}/5    | {tn_count}/5    | "
            f"{false_positives:<15} | {avg_latency:.0f}ms"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
