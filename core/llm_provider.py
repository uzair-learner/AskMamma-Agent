"""Provider abstraction for optional LLM-backed answers, agents, and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import requests
from langchain_core.embeddings import Embeddings

from core import config
from core.observability import configure_langsmith

LLM_UNAVAILABLE_MESSAGE = (
    "LLM is not configured or available. Please configure OpenAI, Azure OpenAI, or Ollama before using AskMamma."
)


class LLMProvider(Protocol):
    name: str

    def generate(self, prompt: str) -> str:
        ...

    def available(self) -> bool:
        ...


class EmbeddingProvider(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def embeddings(self) -> Embeddings:
        ...


def current_provider_name() -> str:
    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        return "Azure OpenAI"
    if config.LLM_PROVIDER == "openai":
        return "OpenAI"
    return "Ollama"


def _ollama_model_catalog() -> list[str]:
    try:
        response = requests.get(f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []
    models = payload.get("models", [])
    return [str(model.get("name", "")).strip() for model in models if model.get("name")]


def resolve_ollama_model_name() -> str:
    configured = config.OLLAMA_MODEL.strip()
    available = _ollama_model_catalog()
    if not available:
        return configured
    if configured in available:
        return configured
    configured_base = configured.split(":", 1)[0]
    for model_name in available:
        if model_name.split(":", 1)[0] == configured_base:
            return model_name
    return available[0]


def current_model_name() -> str:
    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        return config.AZURE_OPENAI_DEPLOYMENT or "Not configured"
    if config.LLM_PROVIDER == "openai":
        return config.OPENAI_MODEL
    return resolve_ollama_model_name()


def ollama_reachable() -> bool:
    return OllamaProvider().available()


@dataclass
class OllamaProvider:
    name: str = "ollama"

    def available(self) -> bool:
        try:
            response = requests.get(f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3)
            return response.ok
        except requests.RequestException:
            return False

    def generate(self, prompt: str) -> str:
        model_name = resolve_ollama_model_name()
        try:
            response = requests.post(
                f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama is configured but not reachable. Start it with `ollama serve` "
                f"and make sure model `{model_name}` is pulled. Error: {exc}"
            ) from exc


@dataclass
class OpenAIProvider:
    name: str = "openai"

    def available(self) -> bool:
        return bool(config.OPENAI_API_KEY)

    def generate(self, prompt: str) -> str:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=config.OPENAI_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)
        return llm.invoke(prompt).content


@dataclass
class AzureOpenAIProvider:
    name: str = "azure"

    def available(self) -> bool:
        return bool(
            config.AZURE_OPENAI_API_KEY
            and config.AZURE_OPENAI_ENDPOINT
            and config.AZURE_OPENAI_DEPLOYMENT
        )

    def generate(self, prompt: str) -> str:
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_API_KEY": config.AZURE_OPENAI_API_KEY,
                "AZURE_OPENAI_ENDPOINT": config.AZURE_OPENAI_ENDPOINT,
                "AZURE_OPENAI_DEPLOYMENT": config.AZURE_OPENAI_DEPLOYMENT,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Azure OpenAI is missing: {', '.join(missing)}")

        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            temperature=0,
        )
        return llm.invoke(prompt).content


@dataclass
class OpenAIEmbeddingProvider:
    name: str = "openai"

    def available(self) -> bool:
        return bool(config.OPENAI_API_KEY)

    def embeddings(self) -> Embeddings:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.OPENAI_EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )


@dataclass
class AzureOpenAIEmbeddingProvider:
    name: str = "azure"

    def available(self) -> bool:
        return bool(
            config.AZURE_OPENAI_API_KEY
            and config.AZURE_OPENAI_ENDPOINT
            and config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        )

    def embeddings(self) -> Embeddings:
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_API_KEY": config.AZURE_OPENAI_API_KEY,
                "AZURE_OPENAI_ENDPOINT": config.AZURE_OPENAI_ENDPOINT,
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Azure OpenAI embeddings are missing: {', '.join(missing)}")

        from langchain_openai import AzureOpenAIEmbeddings

        return AzureOpenAIEmbeddings(
            azure_deployment=config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
        )


def get_llm_provider() -> LLMProvider:
    if config.LLM_PROVIDER == "openai":
        return OpenAIProvider()
    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        return AzureOpenAIProvider()
    return OllamaProvider()


def get_chat_model() -> Any | None:
    """Return a LangChain chat model when tool-calling is supported."""

    configure_langsmith()

    if config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)

    if config.LLM_PROVIDER in {"azure", "azure_openai"} and AzureOpenAIProvider().available():
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            temperature=0,
        )

    if config.LLM_PROVIDER == "ollama" and OllamaProvider().available():
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            return None

        return ChatOllama(model=resolve_ollama_model_name(), temperature=0, base_url=config.OLLAMA_BASE_URL)

    return None


def supports_langchain_agents() -> bool:
    return get_chat_model() is not None


def get_embedding_provider() -> EmbeddingProvider | None:
    if config.LLM_PROVIDER == "openai":
        provider = OpenAIEmbeddingProvider()
        return provider if provider.available() else None

    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        provider = AzureOpenAIEmbeddingProvider()
        return provider if provider.available() else None

    if config.OPENAI_API_KEY:
        return OpenAIEmbeddingProvider()

    azure_provider = AzureOpenAIEmbeddingProvider()
    if azure_provider.available():
        return azure_provider

    return None


def current_runtime_status() -> dict[str, Any]:
    llm_available = supports_langchain_agents()
    return {
        "provider": current_provider_name(),
        "model": current_model_name(),
        "llm_used": llm_available,
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "ollama_reachable": ollama_reachable(),
    }
