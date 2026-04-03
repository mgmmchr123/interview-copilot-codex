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
from icc.core.orchestrator import (
    CLASSIFIER_MODEL,
    CLASSIFIER_PROMPT,
    DEPTH_CODE_MAP,
    DEPTH_PATTERN,
    parse_classifier_output,
)


questions = [
    "Tell me about a time you had to choose between speed and correctness.",
    "How do you handle duplicate messages in a distributed system?",
    "Tell me about a system you built and how it evolved over time.",
    "What is eventual consistency?",
    "Design a rate limiting system for a public API.",
    "Walk me through how you'd debug a Kafka consumer lag issue.",
    "How would you implement a distributed cache?",
    "What's the difference between optimistic and pessimistic locking?",
    "Tell me about a time you had to deal with a production incident.",
    "How do you ensure data consistency across microservices?",
]

MODELS = [
    CLASSIFIER_MODEL,
    "llama-3.3-70b-versatile",
]

CLEAN_FORMAT_RE = re.compile(r"^\[T:[A-Z]+\] \[D:[A-Z]\]$")


def _format_status(is_clean: bool) -> str:
    return "✅ clean" if is_clean else "❌ dirty"


def _extract_intended_depth(raw_output: str) -> str | None:
    depth_match = DEPTH_PATTERN.search(raw_output)
    if not depth_match:
        return None
    val = (depth_match.group(1) or depth_match.group(2)).upper()
    parsed_depth = DEPTH_CODE_MAP.get(val) or val.lower()
    return parsed_depth if parsed_depth in {"basic", "practical", "deep"} else None


def _call_classifier(
    client: OpenAI,
    model: str,
    question: str,
) -> tuple[str, float]:
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

    summary_rows: list[tuple[str, int, float]] = []

    for model in MODELS:
        print(f"=== MODEL: {model} ===")
        clean_count = 0
        latencies: list[float] = []

        for index, question in enumerate(questions, start=1):
            try:
                raw_output, latency_ms = _call_classifier(
                    client=client,
                    model=model,
                    question=question,
                )
                parsed_type, parsed_depth = parse_classifier_output(raw_output)
                is_clean = bool(CLEAN_FORMAT_RE.fullmatch(raw_output))
                intended_depth = _extract_intended_depth(raw_output)
                depth_correct = intended_depth == parsed_depth if intended_depth else False
            except Exception as exc:
                raw_output = f"<ERROR: {exc}>"
                latency_ms = 0.0
                parsed_type, parsed_depth = ("OTHER", "basic")
                is_clean = False
                depth_correct = False

            if is_clean:
                clean_count += 1
            latencies.append(latency_ms)

            print(f'Q{index}: "{question}"')
            print(f"  RAW:    {raw_output!r}")
            print(f"  PARSED: type={parsed_type} depth={parsed_depth}")
            print(
                f"  FORMAT: {_format_status(is_clean)}  |  "
                f"DEPTH_CORRECT: {'✅' if depth_correct else '❌'}  |  "
                f"latency: {latency_ms:.0f}ms"
            )
            print()

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        summary_rows.append((model, clean_count, avg_latency))

    print("=== SUMMARY ===")
    print("Model                    | Compliance | Avg Latency")
    for model, clean_count, avg_latency in summary_rows:
        print(f"{model:<24} | {clean_count}/10       | {avg_latency:.0f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
