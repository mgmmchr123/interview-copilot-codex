import sys
from unittest.mock import MagicMock

sys.path.insert(0, ".")

from icc.core.orchestrator import (
    PHASE_SEQUENCES,
    QUESTION_CLASS_MAP,
    OTHER_TYPE,
    QUESTION_TYPES,
    InterviewOrchestrator,
    PhaseState,
)


# -- 1. PhaseState.start() --------------------------------------------------
for qtype in ["SYSTEM_DESIGN", "DEBUGGING_SCENARIO", "DEEP_DIVE"]:
    ps = PhaseState.start(qtype)
    expected_first = PHASE_SEQUENCES[qtype][0]
    assert ps.phase == expected_first, f"{qtype}: expected {expected_first}, got {ps.phase}"
    assert ps.active is True
    assert ps.phase_version == 0
    assert ps.awaiting_user is True
print("PhaseState.start() OK")


# -- 2. PhaseState.advance() ------------------------------------------------
ps = PhaseState.start("SYSTEM_DESIGN")
phases = PHASE_SEQUENCES["SYSTEM_DESIGN"]
for i, expected in enumerate(phases[1:], start=1):
    ps = PhaseState.advance(ps)
    assert ps.phase == expected, f"step {i}: expected {expected}, got {ps.phase}"
    assert ps.phase_version == i

ps = PhaseState.advance(ps)
assert ps.active is False
print("PhaseState.advance() OK")


# -- 3. QUESTION_CLASS_MAP completeness ------------------------------------
all_types = QUESTION_TYPES | {OTHER_TYPE}
for question_type in all_types:
    assert question_type in QUESTION_CLASS_MAP, f"Missing in QUESTION_CLASS_MAP: {question_type}"
print("QUESTION_CLASS_MAP completeness OK")


# -- 4. detect_override() - user override ----------------------------------
orchestrator = MagicMock()
orch = InterviewOrchestrator.__new__(InterviewOrchestrator)

ps = PhaseState.start("SYSTEM_DESIGN")

for trigger in ["skip", "跳过", "不用了", "直接answer", "next phase", "下一阶段"]:
    result = orch.detect_override(trigger, ps)
    assert result["source"] == "user", f"Failed user override for: {trigger!r}"
print("detect_override() user override OK")


# -- 5. detect_override() - interviewer override ---------------------------
result = orch.detect_override("just design the system", ps)
assert result["source"] == "interviewer", "Failed interviewer override"
assert result["target_phase"] == "HLD"
print("detect_override() interviewer override OK")


# -- 6. detect_override() - no override ------------------------------------
result = orch.detect_override("FR=notifications; Scale=10k QPS", ps)
assert result["source"] == "none"
print("detect_override() no override OK")


# -- 7. Interviewer override blocked when target is earlier phase ----------
ps_at_hld = PhaseState.start("SYSTEM_DESIGN")
ps_at_hld = PhaseState.advance(ps_at_hld)
result = orch.detect_override("just design the system", ps_at_hld)
assert result["source"] == "none", "Should not override to same/earlier phase"
print("detect_override() no backward jump OK")


# -- 8. A-class history retention ------------------------------------------
a_types = [question_type for question_type, question_class in QUESTION_CLASS_MAP.items() if question_class == "A"]
assert set(a_types) == {"SYSTEM_DESIGN", "DEBUGGING_SCENARIO", "DEEP_DIVE"}
print("A-class types correct OK")


# -- 9. _detect_advance() - valid confirmations ----------------------------
ps = PhaseState.start("SYSTEM_DESIGN")
valid_confirmations = [
    "ok", "okay", "yes", "continue", "go ahead",
    "好", "继续", "下一步", "可以", "sounds good",
    "let's go", "got it",
]
for trigger in valid_confirmations:
    result = orch._detect_advance(trigger, ps)
    assert result is True, f"Should advance for: {trigger!r}"
print("_detect_advance() valid confirmations OK")


# -- 10. _detect_advance() - guard: question -------------------------------
non_advances = [
    "what about caching?",
    "FR=notifications; Scale=10k QPS; SLA=P99 300ms",
    "can you explain the HLD more?",
    "skip",
]
for trigger in non_advances:
    result = orch._detect_advance(trigger, ps)
    assert result is False, f"Should NOT advance for: {trigger!r}"
print("_detect_advance() guard conditions OK")


# -- 11. Phase advance sequence via _detect_advance ------------------------
ps = PhaseState.start("SYSTEM_DESIGN")
assert ps.phase == "CLARIFY"
ps = PhaseState.advance(ps)
assert ps.phase == "HLD"
ps = PhaseState.advance(ps)
assert ps.phase == "DEEP_DIVE"
ps = PhaseState.advance(ps)
assert ps.phase == "SCALE_RELIABILITY"
ps = PhaseState.advance(ps)
assert ps.phase == "WRAP"
ps = PhaseState.advance(ps)
assert ps.active is False
print("Full SYSTEM_DESIGN phase sequence OK")


print("\nAll sanity checks passed")
