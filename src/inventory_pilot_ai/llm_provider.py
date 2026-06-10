"""Provider abstraction for optional LLM-backed answers, agents, and embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from importlib.util import find_spec
import logging
from typing import Any, Protocol

import requests
from langchain_core.embeddings import Embeddings

from inventory_pilot_ai import config
from inventory_pilot_ai.observability import configure_langsmith

LLM_UNAVAILABLE_MESSAGE = (
    "LLM is not configured or available. Please configure OpenAI, Azure OpenAI, or Ollama before using Inventory Pilot AI."
)
LOGGER = logging.getLogger(__name__)
_LAST_OLLAMA_ERROR: str | None = None


def _set_last_ollama_error(message: str | None) -> None:
    global _LAST_OLLAMA_ERROR
    _LAST_OLLAMA_ERROR = message


def last_ollama_error() -> str | None:
    return _LAST_OLLAMA_ERROR


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
    # Ollama server exposes model information; try common endpoints and parse safely.
    endpoints = ["/api/models", "/api/tags", "/api/list-models"]
    for ep in endpoints:
        try:
            response = requests.get(f"{config.OLLAMA_BASE_URL.rstrip('/')}{ep}", timeout=3)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            continue

        # payload can be a list of model names, or an object with a "models" key,
        # or an object with "results" containing model entries. Normalize these.
        models = []
        if isinstance(payload, list):
            models = payload
        elif isinstance(payload, dict):
            if "models" in payload and isinstance(payload["models"], list):
                models = payload["models"]
            elif "results" in payload and isinstance(payload["results"], list):
                models = payload["results"]
            else:
                # Some responses return mapping of model name -> metadata
                # e.g. {"llama3.2": {...}}
                maybe_names = [k for k in payload.keys() if isinstance(k, str)]
                if maybe_names:
                    models = maybe_names

        names: list[str] = []
        for entry in models:
            if isinstance(entry, str):
                names.append(entry.strip())
            elif isinstance(entry, dict):
                # Try known keys
                name = entry.get("name") or entry.get("model") or entry.get("id")
                if name:
                    names.append(str(name).strip())

        if names:
            # remove empty and duplicates while preserving order
            seen = set()
            out: list[str] = []
            for n in names:
                if n and n not in seen:
                    seen.add(n)
                    out.append(n)
            return out

    return []


def resolve_ollama_model_name() -> str:
    configured = (config.OLLAMA_MODEL or "").strip()
    return configured


def _ollama_model_available() -> bool:
    configured = (config.OLLAMA_MODEL or "").strip()
    if not configured:
        return False
    return configured in _ollama_model_catalog()


def _ollama_pull_model_name() -> str:
    configured = (config.OLLAMA_MODEL or "").strip()
    return configured.split(":", 1)[0] if configured else configured


def installed_ollama_models() -> list[str]:
    return _ollama_model_catalog()


def validate_ollama_configuration() -> dict[str, Any]:
    configured_model = (config.OLLAMA_MODEL or "").strip()
    base_url = (config.OLLAMA_BASE_URL or "").strip()
    models = installed_ollama_models()
    reachable = bool(models)
    model_available = configured_model in models if configured_model else False
    active_model = configured_model if model_available else None
    error: str | None = None

    if config.LLM_PROVIDER == "ollama":
        LOGGER.info(
            "Loaded Ollama config provider=%s base_url=%s model=%s installed_models=%s",
            config.LLM_PROVIDER,
            base_url,
            configured_model,
            models,
        )
        if not base_url:
            error = "OLLAMA_BASE_URL is not configured."
        elif not reachable:
            error = f"Unable to connect to Ollama at {base_url}"
        elif not model_available:
            error = (
                f"Configured Ollama model '{configured_model}' was not found. "
                f"Installed models: {models}"
            )

    _set_last_ollama_error(error)
    return {
        "configured_model": configured_model,
        "active_model": active_model,
        "ollama_base_url": base_url,
        "ollama_reachable": reachable,
        "installed_ollama_models": models,
        "model_available": model_available,
        "last_error": error,
    }


def current_model_name() -> str:
    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        return config.AZURE_OPENAI_DEPLOYMENT or "Not configured"
    if config.LLM_PROVIDER == "openai":
        return config.OPENAI_MODEL
    return config.OLLAMA_MODEL


def ollama_reachable() -> bool:
    return validate_ollama_configuration()["ollama_reachable"]


@dataclass
class OllamaProvider:
    name: str = "ollama"

    def available(self) -> bool:
        diagnostics = validate_ollama_configuration()
        return diagnostics["ollama_reachable"] and diagnostics["model_available"]

    def generate(self, prompt: str) -> str:
        model_name = resolve_ollama_model_name()
        LOGGER.info(
            "Ollama request selected_model=%s base_url=%s prompt_chars=%s",
            model_name,
            config.OLLAMA_BASE_URL,
            len(prompt),
        )
        try:
            url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
            timeout_seconds = max(5, config.AGENT_TIMEOUT_SECONDS)
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 160,
                },
            }
            response = requests.post(url, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                response_text = str(data.get("response", "")).strip()
                if response_text:
                    return response_text
            raise RuntimeError("Ollama returned an empty response.")
        except requests.RequestException as exc:
            runtime_error = _chat_model_runtime_error()
            if runtime_error:
                LOGGER.error("Ollama request failed selected_model=%s error=%s", model_name, runtime_error)
                raise RuntimeError(runtime_error) from exc
            LOGGER.error(
                "Ollama request failed selected_model=%s raw_error=%s",
                model_name,
                str(exc),
            )
            raise RuntimeError(
                f"Unable to get a response from Ollama model {config.OLLAMA_MODEL} at {config.OLLAMA_BASE_URL}"
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


class LocalHashEmbeddings(Embeddings):
    """Deterministic local embeddings for demo/test RAG without external credentials."""

    dimensions = 128

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class LocalHashEmbeddingProvider:
    name: str = "local-hash"

    def available(self) -> bool:
        return True

    def embeddings(self) -> Embeddings:
        return LocalHashEmbeddings()


def get_llm_provider() -> LLMProvider:
    if config.LLM_PROVIDER == "openai":
        return OpenAIProvider()
    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        return AzureOpenAIProvider()
    return OllamaProvider()


def _dependency_installed(module_name: str) -> bool:
    return find_spec(module_name) is not None


def _chat_model_runtime_error() -> str | None:
    if config.LLM_PROVIDER == "openai":
        if not config.OPENAI_API_KEY:
            return "OPENAI_API_KEY is not configured."
        if not _dependency_installed("langchain_openai"):
            return "langchain-openai package is missing. Run: pip install langchain-openai"
        return None

    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
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
            return f"Azure OpenAI is missing: {', '.join(missing)}"
        if not _dependency_installed("langchain_openai"):
            return "langchain-openai package is missing. Run: pip install langchain-openai"
        return None

    base_url = (config.OLLAMA_BASE_URL or "").strip()
    if not base_url:
        return "OLLAMA_BASE_URL is not configured."
    if not _dependency_installed("langchain_ollama"):
        return "langchain-ollama package is missing. Run: pip install langchain-ollama"

    diagnostics = validate_ollama_configuration()
    models = diagnostics["installed_ollama_models"]
    if not models:
        return f"Unable to connect to Ollama at {base_url}"

    if not diagnostics["model_available"]:
        return f"Ollama model {config.OLLAMA_MODEL} is not available. Run: ollama pull {_ollama_pull_model_name()}"

    return None


def _create_ollama_chat_model() -> Any:
    runtime_error = _chat_model_runtime_error()
    if runtime_error:
        raise RuntimeError(runtime_error)

    from langchain_ollama import ChatOllama

    model_name = resolve_ollama_model_name()
    try:
        return ChatOllama(
            model=model_name,
            temperature=0,
            base_url=config.OLLAMA_BASE_URL,
            num_predict=180,
            request_timeout=max(5, config.AGENT_TIMEOUT_SECONDS),
        )
    except TypeError:
        try:
            return ChatOllama(
                model=model_name,
                base_url=config.OLLAMA_BASE_URL,
                num_predict=180,
                request_timeout=max(5, config.AGENT_TIMEOUT_SECONDS),
            )
        except TypeError:
            try:
                return ChatOllama(
                    model=model_name,
                    temperature=0,
                    num_predict=180,
                    request_timeout=max(5, config.AGENT_TIMEOUT_SECONDS),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Unable to create ChatOllama for model {model_name} at {config.OLLAMA_BASE_URL}. Error: {exc}"
                ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Unable to create ChatOllama for model {model_name} at {config.OLLAMA_BASE_URL}. Error: {exc}"
        ) from exc


def llm_runtime_available() -> bool:
    return _chat_model_runtime_error() is None


def get_chat_model() -> Any | None:
    """Return a LangChain chat model when tool-calling is supported."""

    configure_langsmith()

    runtime_error = _chat_model_runtime_error()
    if runtime_error:
        raise RuntimeError(runtime_error)

    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=config.OPENAI_MODEL, temperature=0, api_key=config.OPENAI_API_KEY)

    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        from langchain_openai import AzureChatOpenAI

        return AzureChatOpenAI(
            azure_deployment=config.AZURE_OPENAI_DEPLOYMENT,
            api_version=config.AZURE_OPENAI_API_VERSION,
            azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            api_key=config.AZURE_OPENAI_API_KEY,
            temperature=0,
        )

    if config.LLM_PROVIDER == "ollama":
        return _create_ollama_chat_model()

    return None


def supports_langchain_agents() -> bool:
    try:
        return get_chat_model() is not None
    except Exception:
        return False


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

    return LocalHashEmbeddingProvider()


def current_runtime_status() -> dict[str, Any]:
    diagnostics = validate_ollama_configuration()
    runtime_error = _chat_model_runtime_error()
    llm_available = runtime_error is None
    return {
        "provider": current_provider_name(),
        "configured_model": diagnostics["configured_model"] or current_model_name(),
        "active_model": diagnostics["active_model"] or (current_model_name() if llm_available else None),
        "model": diagnostics["configured_model"] or current_model_name(),
        "llm_used": llm_available,
        "ollama_base_url": diagnostics["ollama_base_url"],
        "ollama_reachable": diagnostics["ollama_reachable"],
        "installed_ollama_models": diagnostics["installed_ollama_models"],
        "model_available": diagnostics["model_available"],
        "runtime_error": runtime_error,
        "last_error": diagnostics["last_error"],
    }
