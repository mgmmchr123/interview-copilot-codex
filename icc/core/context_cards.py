KAFKA_CARD = """I designed an event-driven workflow using Kafka instead of REST \
because the flow was long-running and needed replay and backpressure handling. \
We persisted workflow state in PostgreSQL and advanced stages via events. \
Consumers are idempotent; failures use exponential backoff → DLQ → replay tool. \
We hit a schema drift bug (producer/consumer mismatch) and fixed it with strict \
validation and schema versioning."""

LATENCY_CARD = """I traced p95/p99 latency and found the DB as the bottleneck. \
Root causes were an N+1 query pattern fetching reference data in a loop, and a \
high-frequency query missing a composite index causing full table scans. \
I replaced the N+1 with a batch query and added a composite index aligned with \
filter conditions; verified via EXPLAIN plans and p95/p99 metrics. \
Result was ~30% latency reduction with significantly lower tail latency variance."""

IDEMPOTENCY_CARD = """I used a client-provided requestId and enforced uniqueness \
in PostgreSQL within the same transaction — either a dedicated idempotency table \
or a unique constraint on the business table. Duplicate detection returns the \
stored result; writes are guarded by DB-level unique constraints to prevent races. \
We had duplicate processing during a consumer rebalance because idempotency was \
only enforced at the API layer. Fix: standardized idempotency at both API and \
consumer layers using DB-level uniqueness, so retries and reprocessing always \
result in exactly one successful write."""

VIRTUAL_THREADS_CARD = """Java 21 Virtual Threads (Project Loom): blocking I/O unmounts \
the carrier thread, allowing it to run other virtual threads. \
ThreadLocal is DANGEROUS with virtual threads: each virtual thread inherits parent \
ThreadLocal state, causing memory leaks at scale (millions of virtual threads = heap \
pressure / leaks). Correct replacement: ScopedValue (JEP 446), immutable and does not \
propagate to child threads by default. \
Synchronized blocks can pin virtual threads to carrier threads — prefer ReentrantLock \
or non-blocking constructs to avoid pinning."""

LATE_EVENT_CARD = """Late-arriving data in stream processing is handled using \
event-time semantics, not arrival time. \
Use watermarks to track how late events can arrive before being considered too late \
to include in a window. Out-of-order events are buffered and merged within the allowed \
lateness window. Late events beyond the watermark are either dropped or sent to a side \
output for correction — do NOT silently discard. \
Kafka log compaction is NOT for ordering or lateness — it is for key-based deduplication \
of the latest value per key. Do NOT confuse it with late-event handling."""

ISOLATION_LEVEL_CARD = """Write Skew occurs when two transactions read overlapping state \
and update different rows, violating a constraint — neither sees the other's write. \
Repeatable Read (MySQL/InnoDB) does NOT prevent Write Skew. \
Snapshot Isolation (SI) alone does NOT prevent Write Skew — do NOT claim SI prevents \
Write Skew, this is a common and fatal interview mistake. \
Only Serializable Snapshot Isolation (SSI), used by PostgreSQL at SERIALIZABLE level, \
detects and prevents Write Skew. Do NOT conflate SI with SSI. \
To prevent Write Skew without full SERIALIZABLE: use SELECT FOR UPDATE on the rows being \
read, or apply application-level version checks (optimistic locking). \
Phantom Reads: prevented by Repeatable Read in MySQL (via gap locks), but NOT by \
Read Committed. \
For high-throughput systems (10k+ TPS): prefer SELECT FOR UPDATE on narrow rows \
(pessimistic) or optimistic locking with version columns, rather than SERIALIZABLE \
which adds overhead at scale."""

CARD_INDEX = {
    "kafka":       KAFKA_CARD,
    "event":       KAFKA_CARD,
    "consumer":    KAFKA_CARD,
    "dlq":         KAFKA_CARD,
    "replay":      KAFKA_CARD,
    "topic":       KAFKA_CARD,
    "latency":     LATENCY_CARD,
    "slow":        LATENCY_CARD,
    "p95":         LATENCY_CARD,
    "p99":         LATENCY_CARD,
    "query":       LATENCY_CARD,
    "index":       LATENCY_CARD,
    "idempotent":       IDEMPOTENCY_CARD,
    "idempotency":      IDEMPOTENCY_CARD,
    "duplicate":        IDEMPOTENCY_CARD,
    "retry":            IDEMPOTENCY_CARD,
    "rebalance":        IDEMPOTENCY_CARD,
    "virtual thread":   VIRTUAL_THREADS_CARD,
    "virtual threads":  VIRTUAL_THREADS_CARD,
    "loom":             VIRTUAL_THREADS_CARD,
    "threadlocal":      VIRTUAL_THREADS_CARD,
    "java 21":              VIRTUAL_THREADS_CARD,
    "late arriving":        LATE_EVENT_CARD,
    "late-arriving":        LATE_EVENT_CARD,
    "out of order":         LATE_EVENT_CARD,
    "event time":           LATE_EVENT_CARD,
    "watermark":            LATE_EVENT_CARD,
    "stream processing":    LATE_EVENT_CARD,
    "write skew":               ISOLATION_LEVEL_CARD,
    "isolation level":          ISOLATION_LEVEL_CARD,
    "repeatable read":          ISOLATION_LEVEL_CARD,
    "read committed":           ISOLATION_LEVEL_CARD,
    "phantom read":             ISOLATION_LEVEL_CARD,
    "phantom":                  ISOLATION_LEVEL_CARD,
    "snapshot isolation":       ISOLATION_LEVEL_CARD,
    "serializable":             ISOLATION_LEVEL_CARD,
    "isolation":                ISOLATION_LEVEL_CARD,
    "transaction isolation":    ISOLATION_LEVEL_CARD,
}


def select_cards(question: str) -> list[str]:
    """Return at most 2 unique relevant cards for the given question."""
    q = question.lower()
    seen = []
    for keyword, card in CARD_INDEX.items():
        if keyword in q and card not in seen:
            seen.append(card)
        if len(seen) == 2:
            break
    return seen
