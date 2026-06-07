"""Run route, tool, answer, and intermediate-step evaluation for AskMamma."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.orchestrator import invoke_agent  # noqa: E402
from scripts.seed_data import seed  # noqa: E402


def evaluate_cases() -> dict[str, object]:
    seed()
    cases = json.loads((ROOT / "data_evaluation.json").read_text(encoding="utf-8"))
    results = []
    session_id = "eval-session"
    categories = {
        "simple_answer": {"passed": 0, "total": 0},
        "rag_quality": {"passed": 0, "total": 0},
        "forecast_quality": {"passed": 0, "total": 0},
    }

    for case in cases:
        result = invoke_agent(case["question"], session_id=session_id)
        expected_tools = set(case["expected_tools"])
        used_tools = set(result["tools_called"])
        route_ok = result["selected_agent"] == case["expected_agent_route"]
        tools_ok = expected_tools.issubset(used_tools)
        answer_ok = all(token.lower() in result["answer"].lower() for token in case.get("answer_must_include", []))
        route_path = result.get("route_path", [])
        steps_ok = all(step in route_path for step in case.get("expected_route_path_contains", []))
        intermediate_ok = bool(result.get("intermediate_steps"))

        ok = route_ok and tools_ok and answer_ok and steps_ok and intermediate_ok
        category = "simple_answer"
        if "DocumentSearchTool" in case["expected_tools"]:
            category = "rag_quality"
        elif "DemoForecastTool" in case["expected_tools"]:
            category = "forecast_quality"
        categories[category]["total"] += 1
        if ok:
            categories[category]["passed"] += 1
        results.append(
            {
                "question": case["question"],
                "passed": ok,
                "category": category,
                "route_ok": route_ok,
                "tools_ok": tools_ok,
                "answer_ok": answer_ok,
                "steps_ok": steps_ok,
                "intermediate_ok": intermediate_ok,
                "selected_agent": result["selected_agent"],
                "tools_called": result["tools_called"],
                "route_path": route_path,
                "answer": result["answer"],
            }
        )

    passed = sum(1 for item in results if item["passed"])
    return {"passed": passed, "total": len(results), "results": results, "categories": categories}


def main() -> None:
    summary = evaluate_cases()
    for item in summary["results"]:
        print(f"{'PASS' if item['passed'] else 'FAIL'} | {item['question']}")
        print(f"  route: {item['selected_agent']}")
        print(f"  tools: {item['tools_called']}")
        print(f"  route path: {item['route_path']}")
        print(f"  answer: {item['answer'][:220]}")
    print(f"\nSummary: {summary['passed']}/{summary['total']} passed")
    raise SystemExit(0 if summary["passed"] == summary["total"] else 1)


if __name__ == "__main__":
    main()
