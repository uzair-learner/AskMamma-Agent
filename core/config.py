"""Application configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


APP_ENV = env("APP_ENV", "development")
DATABASE_URL = env("DATABASE_URL", f"sqlite:///{(ROOT_DIR / 'askmamma.db').as_posix()}")
LLM_PROVIDER = env("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = env("OLLAMA_MODEL", "llama3.2")
OPENAI_MODEL = env("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY = env("OPENAI_API_KEY")
AZURE_OPENAI_API_KEY = env("AZURE_OPENAI_API_KEY") or env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = env("AZURE_OPENAI_DEPLOYMENT") or env("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = env("AZURE_OPENAI_API_VERSION", "2024-10-21")
LANGSMITH_API_KEY = env("LANGSMITH_API_KEY")
LANGSMITH_ENDPOINT = env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = env("LANGSMITH_PROJECT", "askmamma-agent")
LANGSMITH_TRACING = env("LANGSMITH_TRACING", "true").lower() in {"1", "true", "yes", "on"}
VECTOR_STORE_PATH = Path(env("VECTOR_STORE_PATH", str(ROOT_DIR / "vector_store")))
CORS_ORIGINS = [origin.strip() for origin in env("CORS_ORIGINS", "*").split(",") if origin.strip()]
UPLOAD_DIR = ROOT_DIR / "uploads"
REPORT_DIR = ROOT_DIR / "outputs" / "reports"
RATE_LIMIT_PER_MINUTE = int(env("RATE_LIMIT_PER_MINUTE", "30"))
AGENT_MAX_ITERATIONS = int(env("AGENT_MAX_ITERATIONS", "4"))
AGENT_TIMEOUT_SECONDS = int(env("AGENT_TIMEOUT_SECONDS", "45"))
