"""Simple intent classification for supervisor routing."""

from __future__ import annotations


def classify_route(message: str) -> str:
    lowered = message.lower().strip()
    if lowered in {"hi", "hello", "hey", "good morning", "good afternoon"}:
        return "greeting"
    if any(word in lowered for word in ["document", "policy", "manual", "pdf", "docx", "upload", "knowledge base"]):
        return "document"
    if any(word in lowered for word in ["forecast", "demand", "trend", "reorder", "next month", "history"]):
        return "forecast"
    if any(word in lowered for word in ["research", "compare", "benchmark", "best practice", "interview", "architecture", "agent flow", "orchestration"]):
        return "research"
    if any(word in lowered for word in ["report", "summary", "markdown", "json", "txt"]):
        return "report"
    return "inventory"

