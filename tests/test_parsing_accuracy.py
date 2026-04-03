from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from icc.core.orchestrator import parse_classifier_output


TEST_CASES = [
    ('[TYPE: BEHAVIORAL] [DEPTH: practical]',      'BEHAVIORAL',         'practical'),
    ('[CONCEPTUAL:practical]',                     'CONCEPTUAL_DEEP',    'practical'),
    ('[DEEP_DIVE: practical]',                     'DEEP_DIVE',          'practical'),
    ('[TYPE: CONCEPTUAL] [DEPTH: basic]',          'CONCEPTUAL_BASIC',   'basic'),
    ('[SYSTEM_DESIGN] [DEEP]',                     'SYSTEM_DESIGN',      'deep'),
    ('[DEBUGGING_SCENARIO] [DEEP]',                'DEBUGGING_SCENARIO', 'deep'),
    ('[SYSTEM_DESIGN: deep]',                      'SYSTEM_DESIGN',      'deep'),
    ('[TYPE: CONCEPTUAL] [DEPTH: practical]',      'CONCEPTUAL_DEEP',    'practical'),
    ('[TYPE:DEBUGGING_SCENARIO] [DEPTH:practical]','DEBUGGING_SCENARIO', 'practical'),
    ('[TYPES: CONCEPTUAL] [DEPTH: practical]',     'CONCEPTUAL_DEEP',    'practical'),
    ('[TYPE:BEHAVIORAL] [DEPTH:practical]',        'BEHAVIORAL',         'practical'),
    ('[TYPE: SYSTEM_DESIGN] [DEPTH: deep]',        'SYSTEM_DESIGN',      'deep'),
    ('[TYPE:DEEP_DIVE] [DEPTH:practical]',         'DEEP_DIVE',          'practical'),
    ('[TYPE:CONCEPTUAL] [DEPTH:basic]',            'CONCEPTUAL_BASIC',   'basic'),
    ('[TYPE:DEBUGGING_SCENARIO] [DEPTH:practical]','DEBUGGING_SCENARIO', 'practical'),
    ('[TYPE: SYSTEM_DESIGN] [DEPTH: deep]',        'SYSTEM_DESIGN',      'deep'),
    ('[TYPE:CONCEPTUAL] [DEPTH:practical]',        'CONCEPTUAL_DEEP',    'practical'),
]


def main() -> int:
    passed = 0
    failed: list[tuple[str, str, str, str, str]] = []

    print("=== PARSING ACCURACY TEST ===")
    print()

    for raw_output, expected_type, expected_depth in TEST_CASES:
        actual_type, actual_depth = parse_classifier_output(raw_output)
        success = actual_type == expected_type and actual_depth == expected_depth

        if success:
            passed += 1
            print(f"PASS  {raw_output!r}")
            print(f"      -> type={actual_type} depth={actual_depth} \u2705")
        else:
            failed.append(
                (raw_output, actual_type, actual_depth, expected_type, expected_depth)
            )
            print(f"FAIL  {raw_output!r}")
            print(f"      -> type={actual_type} depth={actual_depth} \u274c")
            print(f"      expected: type={expected_type} depth={expected_depth}")
        print()

    total = len(TEST_CASES)
    failed_count = total - passed

    print("=== SUMMARY ===")
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed_count}/{total}")
    print()

    if failed:
        print("Failed cases:")
        for raw_output, actual_type, actual_depth, expected_type, expected_depth in failed:
            details = []
            if actual_type != expected_type:
                details.append(f"type={actual_type}, expected type={expected_type}")
            if actual_depth != expected_depth:
                details.append(f"depth={actual_depth}, expected depth={expected_depth}")
            print(f"  - {raw_output!r} -> got {', '.join(details)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
