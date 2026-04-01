from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from config import AppConfig
from icc.core.orchestrator import InterviewOrchestrator
from icc.core.resume_loader import load_resume
from icc.llm.client import LlmClient


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "benchmark_real_interview.json"
RUNS_PER_CASE = 3
SLEEP_SECONDS = 1
MAX_TOKENS = 200
REAL_INTERVIEW_SCENARIO = "real_interview"
REAL_INTERVIEW_PROMPT = (
    "Can you walk me through how you would design a rate limiting service "
    "that needs to handle 100k requests per second with low latency, and "
    "explain the trade-offs you would make around consistency, availability, "
    "and operational simplicity?"
)

REAL_INTERVIEW_MODELS = [
    {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
]


def build_client(model_cfg: dict[str, Any]) -> OpenAI | None:
    api_key = os.getenv(model_cfg["api_key_env"], "").strip()
    if not api_key:
        print(f"SKIP {model_cfg['model']}: missing {model_cfg['api_key_env']}")
        return None
    kwargs: dict[str, Any] = {"api_key": api_key}
    if model_cfg["base_url"]:
        kwargs["base_url"] = model_cfg["base_url"]
    return OpenAI(**kwargs)


def build_real_interview_messages() -> list[dict[str, Any]]:
    resume_path = SCRIPT_DIR / "resumes" / "betterment.json"
    resume = load_resume(str(resume_path))

    config = AppConfig.from_env()
    orchestrator = InterviewOrchestrator(llm_client=LlmClient(config=config))
    orchestrator.resume_context = resume
    system_message, user_message = orchestrator.build_prompt(
        prompt=REAL_INTERVIEW_PROMPT,
        answer_mode="standard",
        question_type="SYSTEM_DESIGN",
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def extract_delta_text(chunk: Any) -> str:
    if not getattr(chunk, "choices", None):
        return ""
    delta = chunk.choices[0].delta
    content = getattr(delta, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            else:
                text_value = getattr(item, "text", None)
                item_type = getattr(item, "type", None)
                if item_type == "text" and text_value:
                    parts.append(str(text_value))
        return "".join(parts)
    return ""


def run_stream_request(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
) -> dict[str, float | int]:
    started_at = time.perf_counter()
    first_token_at: float | None = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=MAX_TOKENS,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        delta_text = extract_delta_text(chunk)
        if delta_text and first_token_at is None:
            first_token_at = time.perf_counter()

    finished_at = time.perf_counter()
    return {
        "ttft_ms": round(((first_token_at or finished_at) - started_at) * 1000, 2),
        "total_time_ms": round((finished_at - started_at) * 1000, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("\nNo benchmark summary rows to display.")
        return

    headers = [
        "Model",
        "Scenario",
        "Median TTFT (ms)",
        "Median Total (ms)",
        "Median Prompt Tokens",
        "Median Total Tokens",
    ]
    widths = [
        max(len(headers[0]), *(len(str(row["model"])) for row in rows)),
        max(len(headers[1]), *(len(str(row["scenario"])) for row in rows)),
        max(len(headers[2]), *(len(f"{row['ttft_median_ms']:.2f}") for row in rows)),
        max(len(headers[3]), *(len(f"{row['total_time_median_ms']:.2f}") for row in rows)),
        max(len(headers[4]), *(len(str(row["prompt_tokens_median"])) for row in rows)),
        max(len(headers[5]), *(len(str(row["total_tokens_median"])) for row in rows)),
    ]

    def line(values: list[str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    print("\nSummary")
    print(line(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(
            line(
                [
                    str(row["model"]),
                    str(row["scenario"]),
                    f"{row['ttft_median_ms']:.2f}",
                    f"{row['total_time_median_ms']:.2f}",
                    str(row["prompt_tokens_median"]),
                    str(row["total_tokens_median"]),
                ]
            )
        )


def main() -> None:
    messages = build_real_interview_messages()
    per_run_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_cfg in REAL_INTERVIEW_MODELS:
        client = build_client(model_cfg)
        if client is None:
            continue

        model_name = model_cfg["model"]
        run_metrics: list[dict[str, float | int]] = []

        for run_idx in range(1, RUNS_PER_CASE + 1):
            metrics = run_stream_request(client, model_name, messages)
            run_metrics.append(metrics)
            per_run_results.append(
                {
                    "model": model_name,
                    "scenario": REAL_INTERVIEW_SCENARIO,
                    "run": run_idx,
                    **metrics,
                }
            )
            print(
                f"{model_name} | {REAL_INTERVIEW_SCENARIO} | run {run_idx} "
                f"| TTFT={metrics['ttft_ms']:.2f} ms "
                f"| total={metrics['total_time_ms']:.2f} ms "
                f"| prompt_tokens={metrics['prompt_tokens']} "
                f"| completion_tokens={metrics['completion_tokens']} "
                f"| total_tokens={metrics['total_tokens']}"
            )
            time.sleep(SLEEP_SECONDS)

        summary_rows.append(
            {
                "model": model_name,
                "scenario": REAL_INTERVIEW_SCENARIO,
                "ttft_median_ms": statistics.median(
                    float(item["ttft_ms"]) for item in run_metrics
                ),
                "total_time_median_ms": statistics.median(
                    float(item["total_time_ms"]) for item in run_metrics
                ),
                "prompt_tokens_median": int(
                    statistics.median(int(item["prompt_tokens"]) for item in run_metrics)
                ),
                "total_tokens_median": int(
                    statistics.median(int(item["total_tokens"]) for item in run_metrics)
                ),
            }
        )

    print_summary_table(summary_rows)

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "per_run_results": per_run_results,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
