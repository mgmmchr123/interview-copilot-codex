from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from typing import Any

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "quality_results.txt"
MAX_TOKENS = 800

MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

TEST_CASES = [
    {
        "name": "coding_easy",
        "prompt": (
            "Given an array of integers nums and an integer target, return the indices "
            "of the two numbers such that they add up to target. You may assume that "
            "each input would have exactly one solution, and you may not use the same "
            "element twice."
        ),
        "expectation": "hashmap, O(n)",
    },
    {
        "name": "coding_medium",
        "prompt": (
            "Design and implement an LRU Cache with get and put operations in O(1) average time."
        ),
        "expectation": "hashmap + doubly linked list, O(1)",
    },
    {
        "name": "coding_hard",
        "prompt": (
            "Given two sorted arrays nums1 and nums2 of size m and n respectively, "
            "return the median of the two sorted arrays. The overall run time complexity "
            "should be O(log(m+n))."
        ),
        "expectation": "binary search, O(log(m+n))",
    },
    {
        "name": "behavioral",
        "prompt": "Tell me about a time you handled a production incident.",
        "expectation": "clear STAR-style behavioral answer",
    },
    {
        "name": "system_design",
        "prompt": "Design a distributed rate limiter.",
        "expectation": "scalable distributed design with tradeoffs",
    },
]


def build_client() -> OpenAI | None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("Missing GROQ_API_KEY; cannot run quality test.")
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def request_completion(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior backend engineer in a technical interview. "
                    "Answer clearly, completely, and include code when appropriate."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""


def main() -> None:
    client = build_client()
    if client is None:
        return

    sections: list[str] = []

    for model in MODELS:
        for case in TEST_CASES:
            title = f"MODEL: {model}\nCASE: {case['name']}\nEXPECTED: {case['expectation']}"
            divider = "=" * 80
            answer = request_completion(client, model, case["prompt"])

            block = "\n".join(
                [
                    divider,
                    title,
                    "PROMPT:",
                    case["prompt"],
                    "",
                    "ANSWER:",
                    answer,
                    "",
                ]
            )
            print(block)
            sections.append(block)

    RESULTS_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"Saved quality results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
