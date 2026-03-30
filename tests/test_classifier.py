import pytest

from icc.orchestrator import classify_question


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What about handling retries in Kafka consumers?", "follow_up"),
        ("How about if the write path is eventually consistent?", "follow_up"),
        ("Can you elaborate on how you tuned JVM garbage collection?", "follow_up"),
        ("Why is API latency spiking after the latest deployment?", "debugging"),
        ("How do you debug a memory leak in a Spring Boot service?", "debugging"),
        ("What would you check if your Kafka consumers suddenly stop after an error?", "debugging"),
        ("Imagine your primary database goes down during peak traffic. What would you do?", "scenario"),
        ("Suppose you had to migrate a monolith to microservices with zero downtime. How would you approach it?", "scenario"),
        ("Walk me through how you would respond if a regional outage took down one of your AWS availability zones.", "scenario"),
        ("Design a notification service that can send millions of emails per hour.", "system_design"),
        ("Design a rate limiting platform for public APIs.", "system_design"),
        ("Design the architecture for a real-time order tracking platform.", "system_design"),
        ("Tell me about a time you had to resolve a conflict with a product manager.", "behavioral"),
        ("Give me an example of a failure you owned and what you learned from it.", "behavioral"),
        ("Describe a situation where you had to mentor a weaker engineer on your team.", "behavioral"),
        ("Walk me through your implementation of blue-green deployments at your company.", "deep_dive"),
        ("In your current role, how did you improve deployment reliability?", "deep_dive"),
        ("How did you roll out schema changes safely in production?", "deep_dive"),
        ("Why does Kafka guarantee ordering only within a partition?", "conceptual_deep"),
        ("How does the JVM garbage collector work under the hood?", "conceptual_deep"),
        ("Why does optimistic locking in PostgreSQL create tradeoffs in production?", "conceptual_deep"),
        ("What is a deadlock?", "conceptual_basic"),
        ("Explain how a load balancer works.", "conceptual_basic"),
        ("Describe JWT authentication.", "conceptual_basic"),
        ("Implement an LRU cache in Java.", "coding"),
        ("LeetCode-style: solve two sum and state the complexity.", "coding"),
        ("Write a function to detect a cycle in a linked list.", "coding"),
    ],
)
def test_classify_question_by_type(question: str, expected: str) -> None:
    assert classify_question(question) == expected


def test_short_question_with_images_returns_coding() -> None:
    assert classify_question("Explain this diagram", has_images=True) == "coding"


@pytest.mark.xfail(
    strict=False,
    reason="KNOWN_ISSUE: short follow-up phrasing may not be classified consistently.",
)
def test_short_follow_up_without_clear_prefix_known_issue() -> None:
    # KNOWN_ISSUE
    assert classify_question("Caching?") == "follow_up"


def test_deep_dive_with_service_or_system_keyword_known_issue() -> None:
    assert (
        classify_question(
            "Walk me through your service ownership model in your current system."
        )
        == "deep_dive"
    )
