import random

STORIES = [
    {
        "id": "kafka_pipeline",
        "tags": ["kafka", "event", "pipeline", "consumer", "failure",
                 "dlq", "retry", "schema", "drift",
                 "migration", "finra", "workflow", "filing",
                 "regulatory", "transaction", "processing"],
        "summary": (
            "At FINRA I designed a Kafka event-driven pipeline to replace "
            "synchronous REST orchestration for regulatory filing workflows. "
            "The flow had multiple long-running stages: validation, enrichment, "
            "persistence, notifications. Each stage modeled as an event-driven "
            "state transition, persisted in PostgreSQL. Consumers are idempotent "
            "with exponential backoff retries and DLQ fallback. Hit a schema drift "
            "issue where producer added a field but consumer silently defaulted it, "
            "corrupting downstream data. Only caught via data reconciliation. "
            "Fixed with strict schema validation at consumer boundary and enforced "
            "versioning. Built a replay tool to re-publish DLQ messages by error type."
        ),
        "snippets": {
            "opening": [
                "In that Kafka pipeline I built at FINRA...",
                "Going back to that event-driven system...",
                "In that regulatory filing workflow..."
            ],
            "decision": [
                "I chose Kafka over synchronous REST because the workflow was "
                "long-running with multiple stages. With REST, a mid-flow failure "
                "required manual recovery. Kafka gave us durability and replay — "
                "if something failed we could reprocess from the offset safely.",
                "Kafka was a better fit than REST because each filing went through "
                "validation, enrichment, persistence, and notifications. A REST "
                "chain that long breaks badly under failure — Kafka let us model "
                "each step as a recoverable event."
            ],
            "tradeoff": [
                "Kafka gave us replay capability and backpressure handling for "
                "traffic spikes during bulk filings. REST would have overwhelmed "
                "downstream services and required manual intervention on every "
                "mid-flow failure. The operational cost of REST recovery was "
                "the deciding factor.",
                "The core trade-off was operational complexity vs failure safety. "
                "REST is simpler to reason about, but in a long-running workflow "
                "with compliance requirements, we couldn't afford mid-flow failures "
                "that needed manual recovery. Kafka's durability made that acceptable."
            ],
            "mechanism": [
                "Workflow state persisted in PostgreSQL, with each stage advanced "
                "via Kafka events. Consumers are fully idempotent — duplicate "
                "messages produce the same result. Failures hit exponential backoff "
                "retries first, then route to DLQ with error reason and original "
                "payload. A replay tool lets us filter DLQ messages by error type "
                "and re-publish to the original topic.",
                "Each consumer reads an event, processes it, and writes the result "
                "to Postgres in the same transaction that commits the offset. "
                "That's how we guarantee idempotency — if the consumer crashes "
                "mid-write, the offset doesn't advance and the message replays safely."
            ],
            "guarantee": [
                "I guarantee no duplicate DB writes, but I do NOT guarantee "
                "exactly-once delivery. We accept at-least-once and rely on "
                "idempotent consumers to make reprocessing safe. If the consumer "
                "crashes after writing to Postgres but before committing the offset, "
                "the message replays — but the idempotent write means the second "
                "processing produces the same result. The invariant is: no double "
                "write in the DB, even under replay.",
                "I guarantee idempotent state transitions — the same event processed "
                "twice produces the same result in Postgres. I do NOT guarantee "
                "exactly-once delivery from Kafka. That's a deliberate trade-off: "
                "at-least-once with idempotency is operationally simpler and "
                "sufficient for our regulatory filing use case. The cost is that "
                "consumers must be designed to handle replays safely."
            ],
            "invariant": [
                "Yes, the system remains correct. The invariant is: no double "
                "write in the DB, even under replay. If the consumer crashes "
                "after writing to Postgres but before committing the offset, "
                "Kafka redelivers the message. The idempotent write means the "
                "second processing produces the same result. The guarantee holds.",
                "The system is still correct because idempotency is the safety "
                "net. Even if a message is processed twice due to a crash, the "
                "DB state is identical to processing it once. The cost is that "
                "every consumer must be designed with this in mind — but in a "
                "regulatory system, that discipline is non-negotiable."
            ],
            "failure": [
                "We hit a schema drift issue. A producer added a new field but the "
                "consumer was on an older deserialization model. It didn't crash — "
                "the field silently defaulted, corrupting downstream data. "
                "That made it extremely hard to detect.",
                "The nastiest failure was silent data corruption from schema drift. "
                "Producer and consumer were deployed independently, versions drifted, "
                "and a new field just defaulted to null on the consumer side. "
                "No exception, no alert — just wrong data flowing downstream."
            ],
            "detection": [
                "We only found it through data reconciliation, not alerts or logs. "
                "The silent default meant nothing threw an exception. That's what "
                "made it dangerous — the system looked healthy but data was wrong.",
                "It surfaced during a routine reconciliation run that compared "
                "source records against processed output. Counts matched but "
                "field values didn't. That's when we traced it back to the "
                "schema mismatch."
            ],
            "fix": [
                "I added strict schema validation at the consumer boundary and "
                "enforced schema versioning so producer and consumer contracts "
                "are explicit. Any mismatch now fails fast instead of silently defaulting.",
                "We introduced a schema registry and added validation at consumer "
                "startup. If the incoming schema doesn't match the expected version, "
                "the consumer rejects the message immediately and routes to DLQ "
                "instead of silently processing with wrong defaults."
            ],
            "result": [
                "Pipeline processes millions of financial records daily with "
                "zero-loss guarantees. DLQ rate dropped significantly after "
                "schema enforcement. Replay tool gave us full recovery capability "
                "without manual intervention.",
                "After schema enforcement, silent corruption incidents dropped to "
                "zero. The replay tool meant any DLQ spike could be resolved "
                "by ops without engineering involvement."
            ],
            "constraint": [
                "Regulatory environment required every message to be traceable "
                "and replayable for audit purposes. That ruled out any in-memory "
                "orchestration and made Kafka's durability guarantees non-negotiable.",
                "FINRA's audit requirements meant we needed a full event trail. "
                "Every state transition had to be persisted and replayable. "
                "That constraint made Kafka the only realistic option."
            ]
        }
    },
    {
        "id": "latency_improvement",
        "tags": ["latency", "performance", "sql", "query", "index",
                 "n+1", "p95", "p99", "slow", "optimization"],
        "summary": (
            "At FINRA, users reported slow filings. Grafana showed p95/p99 "
            "latency trending up. Used request tracing to isolate the bottleneck "
            "to the database. Found two issues: N+1 query pattern fetching "
            "reference data inside a loop, and a high-frequency query missing "
            "a composite index causing full table scans. Replaced N+1 with batch "
            "query, added composite index aligned with filter conditions. "
            "Validated via EXPLAIN plans and p95/p99 metrics. Result: ~30% "
            "latency reduction with significantly lower tail latency variance."
        ),
        "snippets": {
            "opening": [
                "In that latency investigation at FINRA...",
                "Going back to that performance issue we had...",
                "In that query optimization project..."
            ],
            "decision": [
                "I chose to focus on database-layer optimization over application "
                "caching because tracing showed 80% of request time was in DB "
                "calls. Adding a cache would have masked the problem without "
                "fixing the underlying query inefficiency.",
                "We went straight to the query layer rather than adding a cache "
                "because the data was financial — stale reads weren't acceptable. "
                "Fixing the root cause was the only real option."
            ],
            "tradeoff": [
                "Database-layer fix via index and batch query is permanent and "
                "avoids cache invalidation complexity. Caching would have been "
                "faster to ship but introduced consistency risk for financial "
                "data and stale reads, which we couldn't accept.",
                "The trade-off was delivery speed vs correctness. Cache would "
                "have taken a day to ship but created a whole class of "
                "invalidation problems. The index and batch query fix took "
                "longer but eliminated the problem permanently."
            ],
            "mechanism": [
                "Replaced the N+1 pattern with a single batch query to fetch all "
                "reference data upfront. Added a composite index aligned with the "
                "query's exact filter conditions. Verified index usage via EXPLAIN "
                "plans before and after. Rolled out gradually and tracked p95/p99 "
                "rather than averages to confirm tail latency improvement.",
                "The N+1 was in a loop that fetched one reference record per "
                "filing item. Replaced with a single IN query upfront, then "
                "joined in memory. The index fix required analyzing the WHERE "
                "clause column order and matching the composite index to it exactly."
            ],
            "guarantee": [
                "I guarantee query results reflect current state — no stale reads. "
                "I do NOT guarantee sub-100ms latency under all conditions. "
                "The trade-off is correctness over speed: fixing the root cause "
                "in the DB layer is permanent but took longer than a cache would have. "
                "The cost is that we accepted a short delivery delay to avoid "
                "cache invalidation complexity on financial data.",
                "I guarantee that the fix is permanent — no N+1 queries, "
                "no missing index. I do NOT guarantee this is the last latency "
                "issue we'll see. What I can say is that p99 stabilized "
                "significantly after the fix, which tells us the N+1 was the "
                "primary driver of worst-case latency."
            ],
            "failure": [
                "Two compounding issues: an N+1 query fetching reference data "
                "inside a loop, and a high-frequency query hitting a full table "
                "scan due to a missing composite index. Neither was obvious from "
                "average latency — only visible at p95/p99.",
                "The N+1 was the more dangerous one because it scaled linearly "
                "with filing size. Small filings looked fine. Large bulk filings "
                "hit dozens of extra queries per request, which is why p99 "
                "was so much worse than p50."
            ],
            "detection": [
                "Started with user complaints about slow filings. Checked Grafana "
                "and saw p95/p99 trending up — not alerting yet but degradation "
                "was clear. Used request tracing to break down time per layer and "
                "isolated the bottleneck to specific DB queries.",
                "The key was looking at p95/p99 instead of averages. Average "
                "latency looked acceptable. But p99 was 3x p50, which told us "
                "something was scaling badly with load or data size."
            ],
            "fix": [
                "Replaced the N+1 with a batch query. Added a composite index "
                "matching the query's filter conditions. Confirmed with EXPLAIN "
                "plans that the index was actually being used before releasing.",
                "For the N+1: pulled the reference fetch out of the loop, "
                "batched with IN clause, joined in memory. For the index: "
                "ran EXPLAIN ANALYZE on prod-equivalent data volume to verify "
                "the planner chose the index before we shipped."
            ],
            "result": [
                "~30% reduction in API latency. More importantly, tail latency "
                "variance dropped significantly — p99 became much more stable, "
                "which matters more than average for user-facing filing workflows.",
                "p95 dropped from ~800ms to ~560ms. p99 dropped more dramatically "
                "because the N+1 was the primary driver of worst-case latency. "
                "Bulk filing complaints stopped after the release."
            ],
            "constraint": [
                "Financial data meant we couldn't introduce caching without "
                "solving cache invalidation for consistency. That constraint "
                "pushed us toward fixing the query itself rather than hiding "
                "it behind a cache layer.",
                "Audit requirements meant every query result had to reflect "
                "current state. Eventual consistency from a cache wasn't "
                "acceptable in a regulatory filing context."
            ]
        }
    },
    {
        "id": "technical_disagreement",
        "tags": ["kafka", "rest", "disagreement", "influence", "decision",
                 "pushback", "convince", "alignment", "trade-off",
                 "pipeline", "finra", "naysayer", "resist", "refused",
                 "conflict"],
        "summary": (
            "At FINRA, I proposed moving a filing workflow to Kafka, but a "
            "senior engineer pushed back. He had run the legacy system for "
            "over a decade with almost no incidents, so from his perspective, "
            "introducing Kafka added operational risk without clear upside. "
            "We went back and forth for a few days. He preferred a REST-based "
            "approach because it was easier to debug and reason about. "
            "Instead of arguing technology, I walked through a concrete "
            "failure scenario: if step 3 failed mid-flow, how would we "
            "recover? With REST, we'd need manual reconstruction; with Kafka, "
            "we could replay safely. We didn't fully agree. The compromise "
            "was to move only one workflow to Kafka first, while keeping the "
            "rest unchanged. That reduced the perceived risk. That first "
            "workflow reduced manual recovery incidents noticeably, which "
            "helped build confidence for further migration. What I learned "
            "is that influencing decisions isn't about being right — it's "
            "about reducing perceived risk so others are willing to move."
        ),
        "snippets": {
            "opening": [
                "When we were designing that Kafka pipeline at FINRA...",
                "There was actually pushback on that architecture decision...",
                "That wasn't a straightforward call — there was real debate..."
            ],
            "decision": [
                "I proposed Kafka over REST because the workflow had multiple "
                "long-running steps with partial failure risk. REST would have "
                "required manual recovery or a complex compensation layer every "
                "time something failed mid-flow.",
                "I pushed for Kafka because I could see from our existing "
                "incidents that partial failures in multi-step workflows were "
                "already causing manual intervention. REST would have made "
                "that worse, not better."
            ],
            "tradeoff": [
                "The team's concern was operational overhead — partitions, "
                "offsets, consumer groups. Valid point. My counter was: the "
                "alternative is building a manual recovery layer for REST, "
                "which has its own complexity and is much harder to test. "
                "Kafka's overhead is upfront and well-understood.",
                "REST looked simpler on the surface. But I walked through a "
                "concrete scenario: step 3 of 5 fails — how do we recover? "
                "With Kafka we replay from offset. With REST we rebuild state "
                "manually. That made the trade-off concrete instead of abstract."
            ],
            "mechanism": [
                "I reframed the discussion from 'technology preference' to "
                "'what failure model do we want.' Once the team was evaluating "
                "recovery behavior instead of syntax familiarity, Kafka became "
                "the obvious choice. I also showed bulk filing traffic patterns "
                "— synchronous REST would cascade and overload downstream "
                "services during spikes, while Kafka absorbs with backpressure.",
                "The turning point was walking through a concrete failure "
                "scenario together. Not a hypothetical — an actual case where "
                "a filing partially succeeded and needed recovery. That made "
                "the operational cost of REST visible."
            ],
            "guarantee": [
                "I guarantee the workflow can recover from mid-flow failures "
                "without manual intervention. I do NOT guarantee zero operational "
                "overhead — Kafka adds complexity around partitions, offsets, and "
                "consumer groups. That's the deliberate trade-off: recovery safety "
                "over operational simplicity, which was the right call given "
                "our compliance requirements.",
                "I guarantee replayability — any failed step can be retried safely "
                "from the last committed offset. I do NOT guarantee the senior "
                "engineer's full buy-in from day one. The compromise was migrating "
                "one workflow first, proving the value, then expanding. "
                "That incremental approach was the right call."
            ],
            "failure": [
                "The risk with REST was that partial failures would require "
                "manual intervention or a compensation layer we hadn't built. "
                "That's hidden complexity that shows up in production, not "
                "in design discussions.",
                "If we'd gone with REST, every mid-flow failure would have "
                "needed manual recovery. In a regulatory environment, that's "
                "not acceptable — you can't have engineers manually patching "
                "filing state at 2am."
            ],
            "detection": [
                "I pulled up existing incident logs showing cases where "
                "multi-step workflows had partially succeeded and needed "
                "manual cleanup. That grounded the discussion in real "
                "operational cost, not theoretical risk.",
                "The evidence was already there in our incident history. "
                "I just made it visible and connected it to the architectural "
                "choice we were debating."
            ],
            "fix": [
                "I didn't try to win the argument — I reframed the question. "
                "Instead of 'Kafka vs REST,' I asked 'what failure model do "
                "we want?' That shifted the conversation to trade-offs, and "
                "the team reached the conclusion themselves.",
                "Showing concrete failure scenarios was more effective than "
                "any technical argument. Once people could see the recovery "
                "complexity with REST, the decision became obvious."
            ],
            "result": [
                "We aligned on Kafka for recovery and reliability. The "
                "operational overhead concern didn't go away, but the team "
                "agreed it was the right trade-off given our failure "
                "handling requirements.",
                "The discussion shifted from preference to trade-off analysis, "
                "and we reached a clear decision. More importantly, the team "
                "understood why — which made the implementation smoother."
            ],
            "constraint": [
                "Regulatory environment meant we couldn't accept manual "
                "recovery as a fallback. That constraint made Kafka's "
                "durability guarantees non-negotiable once the team saw "
                "the failure model clearly.",
                "The compliance requirement for auditability and zero-loss "
                "was the deciding constraint. REST with manual recovery "
                "doesn't meet that bar."
            ]
        }
    },
    {
        "id": "decision_mistake",
        "tags": ["schema", "drift", "mistake", "validation", "decision",
                 "wrong", "differently", "lesson", "regret", "hindsight"],
        "summary": (
            "At FINRA we deliberately chose not to enforce strict schema "
            "validation between producers and consumers. The reasoning was "
            "flexibility — teams could evolve message formats without tight "
            "coupling, and we didn't want to slow development with rigid "
            "contracts. We assumed deserialization failures would be loud. "
            "They weren't. A producer added a field, the consumer silently "
            "defaulted it, and data corruption propagated downstream before "
            "we caught it through reconciliation. The lesson: in distributed "
            "financial systems, silent failure is far worse than hard failure. "
            "Flexibility without explicit contracts creates hidden coupling "
            "that's much harder to debug than a fast-fail schema mismatch."
        ),
        "snippets": {
            "opening": [
                "That schema drift incident was actually a decision I made "
                "that I'd do differently now...",
                "If I'm honest, that bug was partly on me...",
                "The schema drift issue came from a call I made early on..."
            ],
            "decision": [
                "I chose not to enforce schema validation upfront because I "
                "wanted to give teams flexibility to evolve their message "
                "formats independently. The thinking was: loose coupling "
                "is good, and strict contracts slow down development.",
                "We deliberately skipped schema contracts early on. I thought "
                "deserialization failures would surface loudly if something "
                "went wrong. That assumption was wrong."
            ],
            "tradeoff": [
                "The trade-off I got wrong was flexibility vs correctness. "
                "I optimized for development speed and loose coupling, but "
                "in a financial data pipeline, silent failure from schema "
                "drift is much worse than a fast-fail contract violation.",
                "I thought tight schema contracts meant slower development. "
                "In hindsight, the cost of debugging silent data corruption "
                "was far higher than the cost of maintaining explicit contracts "
                "from the start."
            ],
            "mechanism": [
                "After the incident, I added strict schema validation at the "
                "consumer boundary and enforced versioning. Any schema mismatch "
                "now fails fast and routes to DLQ instead of silently defaulting. "
                "Hard failure is always preferable to silent corruption in a "
                "system handling financial data.",
                "The fix was schema validation at the consumer boundary — "
                "fail fast on mismatch, route to DLQ, alert immediately. "
                "The opposite of what we had before."
            ],
            "guarantee": [
                "I guarantee that after adding schema validation, any mismatch "
                "fails fast and routes to DLQ immediately. I do NOT guarantee "
                "we caught every edge case before the incident. The lesson is "
                "that in financial systems, silent failure is far worse than "
                "hard failure — I'd rather have a noisy system that fails loudly "
                "than a quiet one that corrupts data.",
                "I guarantee no silent schema drift now — mismatches surface "
                "immediately as DLQ entries with clear error reasons. "
                "I do NOT guarantee we'd have caught this earlier without the "
                "incident. The real fix was changing the design principle: "
                "fail fast always beats silent default in financial data pipelines."
            ],
            "failure": [
                "The producer added a field. The consumer was on an older "
                "model and silently defaulted it to null. No exception, no "
                "alert — just corrupted data flowing downstream. We only "
                "caught it through reconciliation.",
                "The system looked completely healthy. Metrics were normal, "
                "no errors in logs. The corruption was invisible until "
                "reconciliation caught field-level mismatches days later."
            ],
            "detection": [
                "We found it through data reconciliation — comparing source "
                "records against processed output. Record counts matched, "
                "but field values didn't. That's when we traced it back to "
                "the schema mismatch.",
                "Nothing in our monitoring caught it. No alerts, no "
                "exceptions. The reconciliation process was the only safety "
                "net, and it wasn't designed to catch this kind of issue quickly."
            ],
            "fix": [
                "If I were doing it again, I'd enforce schema contracts from "
                "day one. Fail fast on mismatch, route to DLQ, alert "
                "immediately. I'd rather have a noisy system that fails "
                "loudly than a quiet system that corrupts data silently.",
                "The lesson was: flexibility without explicit contracts in "
                "distributed systems creates hidden coupling. That's much "
                "harder to debug than a strict contract that fails fast."
            ],
            "result": [
                "After adding schema validation, silent corruption incidents "
                "dropped to zero. Any schema mismatch now surfaces immediately "
                "as a DLQ entry with a clear error reason.",
                "The system became noisier in a good way — mismatches surface "
                "immediately instead of propagating silently. That's the "
                "right trade-off for financial data."
            ],
            "constraint": [
                "Financial data means silent failure is unacceptable. You can "
                "recover from a hard failure. You often can't recover from "
                "data that looks correct but isn't.",
                "In a regulatory context, data integrity isn't negotiable. "
                "That should have been the constraint that drove schema "
                "enforcement from day one."
            ]
        }
    },
    {
        "id": "influence_decision",
        "tags": ["cache", "latency", "influence", "pushback", "convince",
                 "performance", "trade-off", "evidence", "alignment"],
        "summary": (
            "When we identified the latency issue at FINRA, the initial "
            "suggestion from the team was to add a caching layer — faster "
            "to ship and immediately reduces DB load. I pushed back. Tracing "
            "showed 80% of latency was from an N+1 query and a missing "
            "composite index. Cache would fix symptoms, not root cause. "
            "More importantly, caching financial data introduces stale read "
            "risk and cache invalidation complexity. I made the trade-off "
            "explicit with concrete tracing evidence and query plans. "
            "The team aligned on fixing the DB first, keeping cache as a "
            "fallback. Result: 30% latency reduction with stable p99, "
            "no cache invalidation risk."
        ),
        "snippets": {
            "opening": [
                "There was actually debate on how to fix that latency issue...",
                "The cache suggestion came up early in that investigation...",
                "That wasn't a unanimous call — I had to push back on the "
                "initial approach..."
            ],
            "decision": [
                "I pushed for fixing the DB queries over adding a cache "
                "because tracing showed 80% of latency was from an N+1 "
                "pattern and a missing index. Cache would reduce load but "
                "leave the inefficiency in place.",
                "I chose DB-layer optimization over caching because the "
                "evidence pointed directly at query inefficiency. Caching "
                "on top of a broken query just makes the problem less "
                "visible, not solved."
            ],
            "tradeoff": [
                "Cache was appealing because it's fast to ship. My concern "
                "was stale reads — financial data where a cache returns "
                "outdated state is a real risk, not a theoretical one. "
                "We'd be trading a known performance problem for a potential "
                "data consistency problem.",
                "The team's argument was timeline. Fair point. My counter "
                "was: cache invalidation for financial data is its own "
                "engineering problem, and we'd be building that on top of "
                "queries we know are inefficient. That's compounding risk."
            ],
            "mechanism": [
                "I showed the tracing data directly — request time broken "
                "down by layer, with DB queries clearly dominating. Then "
                "I showed the EXPLAIN plans with full table scans. Once "
                "the team could see where the time was going, the "
                "decision became straightforward.",
                "The turning point was making the evidence concrete. "
                "Not 'I think the DB is the problem' — 'here's the "
                "trace showing 800ms in DB calls, here's the EXPLAIN "
                "plan showing a full table scan on every request.'"
            ],
            "guarantee": [
                "I guarantee the DB fix eliminated the root cause permanently — "
                "no N+1, no missing index. I do NOT guarantee a cache would have "
                "failed — it might have worked. But caching financial data without "
                "solving invalidation is a class of bugs waiting to happen. "
                "The trade-off was delivery speed vs correctness, and correctness "
                "won given the regulatory context.",
            ],
            "failure": [
                "The risk with caching was stale reads on financial data "
                "and cache invalidation complexity. In a regulatory filing "
                "system, serving stale data isn't just a UX problem — "
                "it's a compliance risk.",
                "If we'd gone with cache, we would have masked the N+1 "
                "and missing index. The next time traffic spiked, we'd "
                "hit the same problem — except now we'd also have cache "
                "invalidation logic to debug."
            ],
            "detection": [
                "Request tracing showed the breakdown clearly: 80% of "
                "latency in DB layer, split between the N+1 pattern and "
                "a high-frequency query doing full table scans. That "
                "evidence made the cache argument hard to sustain.",
                "I ran EXPLAIN ANALYZE on the specific queries flagged by "
                "tracing. The missing composite index was obvious once "
                "you looked at the query plan."
            ],
            "fix": [
                "Replaced the N+1 with a batch query. Added a composite "
                "index aligned with filter conditions. Validated via "
                "EXPLAIN plans and p95/p99 metrics before releasing. "
                "Kept cache on the table as a future option if needed.",
                "The DB fix was deterministic — we could predict and "
                "validate the improvement before shipping. Cache would "
                "have been faster to build but harder to validate "
                "for correctness."
            ],
            "result": [
                "30% latency reduction with significantly more stable p99. "
                "No cache invalidation complexity introduced. The team "
                "agreed in retrospect that fixing root cause was the "
                "right call.",
                "p99 stabilized much more than p50 improved — which "
                "told us the N+1 was the primary driver of worst-case "
                "latency. Cache would not have caught that."
            ],
            "constraint": [
                "Financial data meant stale reads weren't acceptable. "
                "That constraint ruled out caching as a primary solution "
                "and pushed us toward fixing the query itself.",
                "Audit requirements meant every query result had to "
                "reflect current state. That single constraint made "
                "the cache argument collapse once I stated it explicitly."
            ]
        }
    },
    {
        "id": "mentoring",
        "tags": ["mentor", "mentoring", "junior", "engineer", "growth",
                 "debugging", "teach", "guide", "develop", "coaching"],
        "summary": (
            "At FINRA I mentored a junior engineer who was struggling with "
            "production debugging. His pattern was jumping straight to fixes "
            "— restarting services, adding logs — without understanding root "
            "cause. Instead of giving answers, I changed how we worked together: "
            "when issues came up, I asked him to walk me through his signals, "
            "hypotheses, and expected outcomes before touching anything. I "
            "introduced tracing and metrics as tools to break down where time "
            "was actually going. The turning point was a latency spike where "
            "he wanted to add caching — I guided him to use tracing first, "
            "which showed an N+1 query as the real cause. Over time his "
            "debugging became hypothesis-driven instead of reactive. The lesson: "
            "mentoring isn't transferring knowledge, it's helping someone build "
            "a mental model for how systems behave."
        ),
        "snippets": {
            "opening": [
                "There's one mentoring experience that stands out at FINRA...",
                "I had a junior engineer on the team who changed how I think "
                "about mentoring...",
                "One of the more impactful things I did at FINRA wasn't "
                "technical — it was mentoring..."
            ],
            "decision": [
                "I decided not to give answers directly. Instead I changed "
                "how we worked through problems together — asking him to "
                "walk me through his signals and hypotheses before touching "
                "anything. The goal was to build a debugging mental model, "
                "not transfer specific solutions.",
                "I chose to guide through questions rather than answers. "
                "Every time an issue came up, I'd ask: what signals do you "
                "see, what's your hypothesis, what do you expect to happen "
                "if you're right? That process mattered more than any "
                "individual fix."
            ],
            "tradeoff": [
                "Giving answers is faster in the moment but doesn't scale. "
                "The investment in slowing down and asking questions paid "
                "off when he started debugging independently — without "
                "needing me to validate every step.",
                "Short-term, my approach meant issues took longer to resolve "
                "together. Long-term, it meant I stopped being a bottleneck "
                "in his debugging process entirely."
            ],
            "mechanism": [
                "I introduced tracing and metrics as the first step in any "
                "investigation — break down where time is actually going "
                "before forming hypotheses. The turning point was a latency "
                "spike where he wanted to add caching. I asked him to run "
                "tracing first. It showed an N+1 query. He fixed the root "
                "cause instead of masking it.",
                "The framework I gave him was: signals first, hypothesis "
                "second, action third. Never skip to action without the "
                "first two. Tracing and metrics were the tools that made "
                "signals concrete instead of guesswork."
            ],
            "guarantee": [
                "I guarantee the mentoring approach built a lasting debugging "
                "mental model — he became significantly more independent over time. "
                "I do NOT guarantee this works for every engineer or every timeline. "
                "The trade-off is short-term speed: issues took longer to resolve "
                "together, but that investment paid off when he stopped needing "
                "validation for every step.",
            ],
            "failure": [
                "His original pattern was fixing symptoms — restarting "
                "services, adding logs — without root cause analysis. That "
                "works for simple issues but breaks down as systems get "
                "more complex. It also creates dependency on whoever knows "
                "the system best.",
                "The risk of not mentoring this way is you create an "
                "engineer who can only fix problems they've seen before. "
                "In a complex distributed system, that's not enough."
            ],
            "detection": [
                "I noticed the pattern early — he'd come to me with a fix "
                "already in mind, not a diagnosis. That told me he was "
                "skipping the investigation phase and going straight to "
                "action based on instinct.",
                "The signal was that his fixes worked inconsistently. "
                "Sometimes he'd get lucky, sometimes not. That inconsistency "
                "was a sign he wasn't understanding root cause — just "
                "pattern-matching to previous solutions."
            ],
            "fix": [
                "Changed the dynamic from 'come to me with a problem, "
                "leave with a solution' to 'come to me with your "
                "investigation, we'll find the gap together.' That shift "
                "made him more self-sufficient over time.",
                "Structured every debugging session the same way: what "
                "do you see, what do you think is happening, what would "
                "prove or disprove that? Repetition built the habit."
            ],
            "result": [
                "Over time his debugging became hypothesis-driven and "
                "structured. He stopped jumping to fixes and started "
                "asking 'what does the data say?' The N+1 case was the "
                "clearest turning point — he found the root cause himself "
                "using tracing, without prompting.",
                "He became significantly more independent. The frequency "
                "of him coming to me for validation dropped noticeably, "
                "and when he did come, it was with a clear diagnosis "
                "rather than a symptom."
            ],
            "constraint": [
                "Production environment meant we couldn't use real incidents "
                "as pure learning exercises — issues needed to get resolved. "
                "That tension between 'fix it fast' and 'teach the process' "
                "was the constraint I had to balance throughout.",
                "Time pressure from production issues made the slow-down "
                "approach feel risky. But I kept it because the alternative "
                "— staying the bottleneck for every debugging session — "
                "wasn't sustainable."
            ]
        }
    },
    {
        "id": "prioritization",
        "tags": ["prioritize", "prioritization", "multiple", "urgent",
                 "conflict", "deadline", "trade-off", "stakeholder",
                 "pressure", "decision", "feature", "production"],
        "summary": (
            "At FINRA we had a latency issue affecting user filings in "
            "production at the same time as pressure to deliver a feature "
            "for a regulatory deadline. Both were important but had different "
            "risk profiles. I framed the decision by impact: the latency "
            "issue was actively degrading in production, while the feature "
            "had more runway. I made the call to stabilize production first, "
            "communicated the trade-off explicitly to stakeholders — if the "
            "system wasn't reliable, the feature wouldn't matter. I broke "
            "the latency fix into chunks for quick measurable progress and "
            "kept feature work unblocked where possible. Once stable, we "
            "shifted back to feature delivery. The lesson: prioritization "
            "is about impact and risk, not just urgency — and making "
            "those trade-offs explicit so everyone is aligned."
        ),
        "snippets": {
            "opening": [
                "There was a period at FINRA where I had to make a hard "
                "prioritization call...",
                "One situation that tested my judgment was when two urgent "
                "things landed at the same time...",
                "At FINRA I had a production issue and a feature deadline "
                "collide at the same time..."
            ],
            "decision": [
                "I made the call to prioritize the production latency issue "
                "over the feature deadline. The reasoning was risk profile: "
                "the latency issue was actively degrading and had no floor, "
                "while the feature had more runway before it became critical.",
                "I chose to stabilize production first. Not because the "
                "feature didn't matter, but because a degrading system "
                "makes everything else harder — including the feature "
                "we were trying to ship."
            ],
            "tradeoff": [
                "The trade-off was reliability now vs feature delivery "
                "on schedule. I made it explicit to stakeholders: if we "
                "don't fix the latency issue, the new feature ships into "
                "an unreliable system. That reframe helped them understand "
                "why production stability was the right first priority.",
                "Urgency and importance aren't the same thing. The feature "
                "felt urgent because of the deadline, but the production "
                "issue was more important because it had an unbounded "
                "downside. I made that distinction explicit."
            ],
            "mechanism": [
                "I broke the latency fix into the smallest chunks that "
                "would show measurable progress — query optimization first, "
                "index second — so we could ship partial improvements "
                "quickly. That reduced the blast radius of the deadline "
                "conflict. At the same time I kept feature work unblocked "
                "where it didn't compete for the same engineering time.",
                "Communicated the trade-off proactively rather than waiting "
                "for stakeholders to ask. Explained the risk profile, the "
                "decision, and the sequencing. That gave everyone visibility "
                "instead of uncertainty."
            ],
            "guarantee": [
                "I guarantee the trade-off was communicated explicitly before "
                "anyone had to ask. I do NOT guarantee everyone agreed with the "
                "decision initially. The cost was a short feature delay, but "
                "shipping into a degraded system would have been worse. "
                "Making the risk profile visible early is what kept stakeholders "
                "aligned throughout.",
            ],
            "failure": [
                "The risk of not prioritizing explicitly was that we'd "
                "try to do both, do neither well, and ship a feature "
                "into a degraded system. That's worse than a short delay "
                "on either one.",
                "Reacting to whoever was loudest would have meant "
                "context-switching constantly and making no real progress "
                "on either problem. The latency issue would have kept "
                "degrading while the feature also slipped."
            ],
            "detection": [
                "The conflict surfaced when the latency issue started "
                "generating user complaints while the feature deadline "
                "was fixed and visible. Both had legitimate stakeholders "
                "pushing for their priority.",
                "I recognized the conflict early enough to frame it "
                "proactively — before it became a fire drill. That gave "
                "me time to communicate the trade-off clearly instead "
                "of just reacting."
            ],
            "fix": [
                "Stabilized the latency issue first using the chunked "
                "approach — measurable progress fast. Then shifted "
                "engineering focus back to feature delivery once the "
                "system was stable. Feature shipped slightly later but "
                "into a reliable system.",
                "The key move was making the trade-off explicit to "
                "stakeholders before they had to ask. That built trust "
                "in the decision even when the outcome wasn't what "
                "everyone originally wanted."
            ],
            "result": [
                "Production stabilized, latency issue resolved. Feature "
                "shipped with a short delay but into a reliable system. "
                "Stakeholders were aligned throughout because the "
                "trade-off was visible from the start.",
                "No one was surprised by the delay because the reasoning "
                "was communicated early. That's the difference between "
                "a delay that erodes trust and one that doesn't."
            ],
            "constraint": [
                "Regulatory deadline meant the feature couldn't slip "
                "indefinitely — that constraint was real. But an "
                "unreliable system at deadline is worse than a short "
                "slip. That's the constraint I weighed explicitly.",
                "Both timelines had external stakeholders — users "
                "affected by latency and regulators expecting the "
                "feature. Making the prioritization explicit meant "
                "neither group was left guessing."
            ]
        }
    },
]

# Snippet routing: maps follow-up keywords to snippet keys
SNIPPET_ROUTING = {
    "failure":   ["what failed", "what went wrong", "what broke",
                  "incident", "issue", "bug", "problem",
                  "hardest", "difficult", "toughest", "worst",
                  "what if", "crashes", "crash", "consumer crash",
                  "what happens if", "goes down", "fails",
                  "if the consumer", "if it crashes", "if it fails"],
    "detection": ["how did you find", "how did you detect",
                  "how did you know", "how did you discover"],
    "fix":       ["how did you fix", "what did you change",
                  "how did you solve", "what was the fix"],
    "decision":  ["why did you choose", "why kafka", "why not",
                  "why did you pick", "why that approach"],
    "tradeoff":  ["what did you consider", "trade-off", "tradeoff",
                  "what were the alternatives", "why not just"],
    "mechanism": ["how does it work", "how does that work",
                  "how is it implemented", "how does the system"],
    "guarantee": [
        "ensure consistency", "how do you ensure",
        "guarantee consistency", "how do you guarantee",
        "how do you maintain", "maintain consistency",
    ],
    "invariant": [
        "still correct", "remain correct", "system correct",
        "still safe", "still valid", "still consistent",
        "after that", "after the crash", "after failure",
        "still work", "still holds", "invariant",
        "why safe", "why correct", "why still",
    ],
    "result":    ["what was the result", "what was the outcome",
                  "what happened after", "did it work"],
    "constraint":["what was the constraint", "compliance", "audit",
                  "what forced", "why were you required"]
}

EXPERIENCE_KEYWORDS = [
    "what failed", "what went wrong", "what broke", "what bug",
    "how did you find", "how did you detect", "how did you discover",
    "how did you fix", "what did you change", "what was the result",
    "what happened", "incident", "tell me more about",
    "hardest bug", "hardest issue", "most difficult", "toughest",
    "hardest part",
    "ensure consistency", "how do you ensure", "guarantee consistency",
    "still correct", "still safe", "still consistent",
    "why safe", "why correct", "after the crash",
]


IMPORTANT_TAGS = {
    "kafka", "pipeline", "migration", "workflow", "filing",
    "latency", "mentor", "prioritize", "disagree", "schema",
}

TAG_SYNONYMS = {
    "migration": ["move", "migrate", "transition", "migrating"],
    "workflow":  ["flow", "process"],
    "failure":   ["crash", "broke", "failed", "incident", "bug"],
    "latency":   ["slow", "performance", "speed", "p95", "p99"],
    "mentor":    ["junior", "coaching", "guide", "teach"],
    "disagree":  ["pushback", "conflict", "convince", "influence"],
}


def _tag_matches(tag: str, q: str) -> bool:
    """Check if a tag or any of its synonyms appears in the query."""
    if tag in q:
        return True
    for synonym in TAG_SYNONYMS.get(tag, []):
        if synonym in q:
            return True
    return False


def find_story(question: str) -> dict | None:
    """Return the best matching story or None."""
    q = question.lower()
    best_story = None
    best_score = 0
    for story in STORIES:
        score = 0
        for tag in story["tags"]:
            if _tag_matches(tag, q):
                score += 2 if tag in IMPORTANT_TAGS else 1
        if score > best_score:
            best_score = score
            best_story = story
    return best_story if best_score > 0 else None


def is_experience_followup(question: str) -> bool:
    """True if the follow-up asks about a specific experience/event."""
    q = question.lower()
    return any(kw in q for kw in EXPERIENCE_KEYWORDS)


def get_snippet(story: dict, question: str) -> str | None:
    """
    Return the best matching snippet for a follow-up question.
    Picks a random variant from the matching list for naturalness.
    Returns opening + snippet if match found, else None.
    """
    q = question.lower()
    for snippet_key, keywords in SNIPPET_ROUTING.items():
        if any(kw in q for kw in keywords):
            snippets = story["snippets"].get(snippet_key, [])
            if snippets:
                opening = random.choice(story["snippets"]["opening"])
                content = random.choice(snippets)
                return f"{opening} {content}"

    # Single-word fallback for failure routing
    if "hardest" in q or "difficult" in q or "toughest" in q:
        snippets = story["snippets"].get("failure", [])
        if snippets:
            opening = random.choice(story["snippets"]["opening"])
            content = random.choice(snippets)
            return f"{opening} {content}"

    return None


def get_summary(story: dict) -> str:
    """Return the full story summary for LLM context injection."""
    return story["summary"]
