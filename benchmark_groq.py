from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import base64
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "benchmark_results.json"
IMAGE_PATH = SCRIPT_DIR / "test_leetcode.png"
RUNS_PER_CASE = 3
SLEEP_SECONDS = 1
MAX_TOKENS = 200

TEXT_CASES = {
    "behavioral": "Tell me about a time you disagreed with your manager and how you handled it.",
    "coding_text": "Given an array of integers, find two numbers that add up to a target. Return their indices.",
    "system_design": "Design a URL shortener like bit.ly that handles 100 million requests per day.",
}
IMAGE_CASE_NAME = "coding_image"
IMAGE_CASE_PROMPT = "solve this problem"

TEXT_MODELS = [
    {
        "provider": "groq",
        "model": "openai/gpt-oss-20b",
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
    },
    {
        "provider": "groq",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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

IMAGE_MODELS = [
    {
        "provider": "groq",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
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


def encode_image_data_url(path: Path) -> str:
    with Image.open(path) as img:
        image_format = (img.format or "PNG").lower()
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/{image_format};base64,{b64}"


def build_messages(prompt: str, image_data_url: str | None = None) -> list[dict[str, Any]]:
    if image_data_url is None:
        user_content: Any = prompt
    else:
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]

    return [
        {
            "role": "system",
            "content": "You are a senior backend engineer. Answer clearly and directly.",
        },
        {"role": "user", "content": user_content},
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
    prompt: str,
    image_data_url: str | None = None,
) -> dict[str, float | int]:
    started_at = time.perf_counter()
    first_token_at: float | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=build_messages(prompt, image_data_url=image_data_url),
        stream=True,
        max_tokens=MAX_TOKENS,
    )

    for chunk in stream:
        delta_text = extract_delta_text(chunk)
        if delta_text and first_token_at is None:
            first_token_at = time.perf_counter()

    finished_at = time.perf_counter()
    return {
        "ttft_ms": round(((first_token_at or finished_at) - started_at) * 1000, 2),
        "total_time_ms": round((finished_at - started_at) * 1000, 2),
    }


def print_summary_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("\nNo benchmark summary rows to display.")
        return

    headers = ["Model", "Scenario", "TTFT中位数(ms)", "总时间中位数(ms)"]
    widths = [
        max(len(headers[0]), *(len(str(row["model"])) for row in rows)),
        max(len(headers[1]), *(len(str(row["scenario"])) for row in rows)),
        max(len(headers[2]), *(len(f"{row['ttft_median_ms']:.2f}") for row in rows)),
        max(len(headers[3]), *(len(f"{row['total_time_median_ms']:.2f}") for row in rows)),
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
                ]
            )
        )


def main() -> None:
    image_data_url: str | None = None
    if IMAGE_PATH.exists():
        image_data_url = encode_image_data_url(IMAGE_PATH)
    else:
        print(f"SKIP {IMAGE_CASE_NAME}: image file not found at {IMAGE_PATH}")

    per_run_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_cfg in TEXT_MODELS:
        client = build_client(model_cfg)
        if client is None:
            continue

        model_name = model_cfg["model"]

        for scenario, prompt in TEXT_CASES.items():
            run_metrics: list[dict[str, float | int]] = []
            for run_idx in range(1, RUNS_PER_CASE + 1):
                metrics = run_stream_request(client, model_name, prompt)
                run_metrics.append(metrics)
                per_run_results.append(
                    {
                        "model": model_name,
                        "scenario": scenario,
                        "run": run_idx,
                        **metrics,
                    }
                )
                print(
                    f"{model_name} | {scenario} | run {run_idx} "
                    f"| TTFT={metrics['ttft_ms']:.2f} ms "
                    f"| total={metrics['total_time_ms']:.2f} ms"
                )
                time.sleep(SLEEP_SECONDS)

            summary_rows.append(
                {
                    "model": model_name,
                    "scenario": scenario,
                    "ttft_median_ms": statistics.median(
                        float(item["ttft_ms"]) for item in run_metrics
                    ),
                    "total_time_median_ms": statistics.median(
                        float(item["total_time_ms"]) for item in run_metrics
                    ),
                }
            )

    if image_data_url is not None:
        for model_cfg in IMAGE_MODELS:
            client = build_client(model_cfg)
            if client is None:
                continue

            model_name = model_cfg["model"]
            run_metrics = []
            for run_idx in range(1, RUNS_PER_CASE + 1):
                metrics = run_stream_request(
                    client,
                    model_name,
                    IMAGE_CASE_PROMPT,
                    image_data_url=image_data_url,
                )
                run_metrics.append(metrics)
                per_run_results.append(
                    {
                        "model": model_name,
                        "scenario": IMAGE_CASE_NAME,
                        "run": run_idx,
                        **metrics,
                    }
                )
                print(
                    f"{model_name} | {IMAGE_CASE_NAME} | run {run_idx} "
                    f"| TTFT={metrics['ttft_ms']:.2f} ms "
                    f"| total={metrics['total_time_ms']:.2f} ms"
                )
                time.sleep(SLEEP_SECONDS)

            summary_rows.append(
                {
                    "model": model_name,
                    "scenario": IMAGE_CASE_NAME,
                    "ttft_median_ms": statistics.median(
                        float(item["ttft_ms"]) for item in run_metrics
                    ),
                    "total_time_median_ms": statistics.median(
                        float(item["total_time_ms"]) for item in run_metrics
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
