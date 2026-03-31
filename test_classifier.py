from dotenv import load_dotenv

load_dotenv()

import time

from config import AppConfig
from icc.llm.client import LlmClient
from icc.orchestrator import InterviewOrchestrator


config = AppConfig.from_env()
client = LlmClient(config=config)
orch = InterviewOrchestrator(client)

tests = [
    ("Given a binary tree return level order traversal", False),
    ("Tell me about a time you disagreed with your manager", False),
    ("Design a distributed rate limiter", False),
    ("Your Kafka consumer lag keeps growing what do you do", False),
    ("What is the difference between a process and a thread", False),
]

for transcript, has_image in tests:
    start = time.perf_counter()
    result = orch._classify_question(transcript, has_image)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"{elapsed:6.0f}ms  {result:25s}  {transcript[:50]}")
