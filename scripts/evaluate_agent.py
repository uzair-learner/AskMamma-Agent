"""Run lightweight route/tool evaluation for the inventory agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import invoke_agent  # noqa: E402
from scripts.seed_data import seed  # noqa: E402


def main() -> None:
    seed()
    cases = json.loads((ROOT / "data_evaluation.json").read_text(encoding="utf-8"))
    passed = 0
    session_id = "eval-session"
    for case in cases:
        result = invoke_agent(case["question"], session_id=session_id)
        expected_tools = set(case["expected_tools"])
        used_tools = set(result["tools_called"])
        route_ok = result["selected_agent"] == case["expected_agent_route"]
        tools_ok = expected_tools.issubset(used_tools)
        ok = route_ok and tools_ok
        passed += int(ok)
        print(f"{'PASS' if ok else 'FAIL'} | {case['question']}")
        print(f"  route: {result['selected_agent']} expected {case['expected_agent_route']}")
        print(f"  tools: {result['tools_called']} expected {case['expected_tools']}")
        print(f"  answer: {result['answer'][:180]}")
    print(f"\nSummary: {passed}/{len(cases)} passed")
    raise SystemExit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
