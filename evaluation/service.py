"""Evaluation helpers for routing, retrieval, and tool quality."""

from __future__ import annotations

from typing import Any


def summarize_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    total = result.get("total", 0)
    passed = result.get("passed", 0)
    score = 0 if total == 0 else round((passed / total) * 100, 2)
    return {
        "total_cases": total,
        "passed_cases": passed,
        "score_percent": score,
        "categories": result.get("categories", {}),
    }
