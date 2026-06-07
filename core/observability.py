"""Tracing and redaction helpers for local and LangSmith-backed runs."""

from __future__ import annotations

import os
from typing import Any

from core import config


SENSITIVE_VALUES = [
    config.OPENAI_API_KEY,
    config.AZURE_OPENAI_API_KEY,
    config.LANGSMITH_API_KEY,
]


def configure_langsmith() -> bool:
    """Enable LangSmith tracing via environment variables when configured."""

    enabled = bool(config.LANGSMITH_API_KEY and config.LANGSMITH_TRACING)
    if not enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ.pop("LANGSMITH_API_KEY", None)
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGSMITH_ENDPOINT"] = config.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = config.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT
    return True


def tracing_backend_name() -> str:
    return "langsmith" if configure_langsmith() else "sqlite"


def redact_text(value: str) -> str:
    redacted = value
    for secret in SENSITIVE_VALUES:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def redact_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return redact_text(payload)
    if isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {key: redact_payload(value) for key, value in payload.items()}
    return payload


def safe_error_message(exc: Exception) -> str:
    return redact_text(str(exc))
