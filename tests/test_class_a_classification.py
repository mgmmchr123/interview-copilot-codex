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


DEEP_DIVE_QUESTIONS = [
    "Tell me about a system you built and how it evolved over time.",
    "Walk me through a project where you had to scale a service.",
    "Tell me about the most complex system you've worked on.",
    "Describe a system you designed from scratch.",
    "Tell me about a time you had to refactor a large codebase.",
]

DEBUGGING_QUESTIONS = [
    "Walk me through how you'd debug a Kafka consumer lag issue.",
    "Your service latency suddenly spiked in production, what do you do?",
    "How would you diagnose a memory leak in a Java service?",
    "Your API error rate just jumped to 10%, how do you investigate?",
    "A downstream service is returning 503s intermittently, walk me through your approach.",
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


def _run_bucket(
    client: OpenAI,
    model: str,
    label: str,
    questions: list[str],
    expected_type: str,
) -> tuple[int, list[float]]:
    print(f"--- {label} ---")
    correct_count = 0
    latencies: list[float] = []

    for index, question in enumerate(questions, start=1):
        try:
            raw_output, latency_ms = _call_classifier(client, model, question)
            parsed_type, parsed_depth = parse_classifier_output(raw_output)
            is_correct = parsed_type == expected_type
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

    summary_rows: list[tuple[str, int, int, float]] = []

    for model in MODELS:
        print(f"=== MODEL: {model} ===")
        print()

        deep_dive_correct, deep_dive_latencies = _run_bucket(
            client=client,
            model=model,
            label="DEEP_DIVE",
            questions=DEEP_DIVE_QUESTIONS,
            expected_type="DEEP_DIVE",
        )

        debugging_correct, debugging_latencies = _run_bucket(
            client=client,
            model=model,
            label="DEBUGGING_SCENARIO",
            questions=DEBUGGING_QUESTIONS,
            expected_type="DEBUGGING_SCENARIO",
        )

        all_latencies = deep_dive_latencies + debugging_latencies
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
        summary_rows.append(
            (model, deep_dive_correct, debugging_correct, avg_latency)
        )

    print("=== SUMMARY ===")
    print("Model                    | DEEP_DIVE  | DEBUGGING  | Avg Latency")
    for model, deep_dive_correct, debugging_correct, avg_latency in summary_rows:
        print(
            f"{model:<24} | {deep_dive_correct}/5        | "
            f"{debugging_correct}/5        | {avg_latency:.0f}ms"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
