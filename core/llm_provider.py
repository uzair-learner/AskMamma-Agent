"""Provider abstraction for optional LLM-backed answers, agents, and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Protocol

import requests
from langchain_core.embeddings import Embeddings

from core import config
from core.observability import configure_langsmith

LLM_UNAVAILABLE_MESSAGE = (
    "LLM is not configured or available. Please configure OpenAI, Azure OpenAI, or Ollama before using Inventory Pilot AI."
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
    available = _ollama_model_catalog()
    if not available:
        return configured
    if configured in available:
        return configured
    configured_base = configured.split(":", 1)[0]
    for model_name in available:
        if model_name.split(":", 1)[0] == configured_base:
            return model_name
    return configured


def _ollama_model_available() -> bool:
    configured = (config.OLLAMA_MODEL or "").strip()
    if not configured:
        return False
    resolved = resolve_ollama_model_name()
    if not resolved:
        return False
    configured_base = configured.split(":", 1)[0]
    resolved_base = resolved.split(":", 1)[0]
    return configured == resolved or configured_base == resolved_base


def _ollama_pull_model_name() -> str:
    configured = (config.OLLAMA_MODEL or "").strip()
    return configured.split(":", 1)[0] if configured else configured


def current_model_name() -> str:
    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        return config.AZURE_OPENAI_DEPLOYMENT or "Not configured"
    if config.LLM_PROVIDER == "openai":
        return config.OPENAI_MODEL
    return config.OLLAMA_MODEL


def ollama_reachable() -> bool:
    return OllamaProvider().available()


@dataclass
class OllamaProvider:
    name: str = "ollama"

    def available(self) -> bool:
        # Check base URL and whether configured model appears in the server catalog.
        base = (config.OLLAMA_BASE_URL or "").strip()
        if not base:
            return False
        try:
            models = _ollama_model_catalog()
            if not models:
                # try a simple health check
                try:
                    resp = requests.get(f"{base.rstrip('/')}/ping", timeout=2)
                    if resp.ok:
                        return True
                except requests.RequestException:
                    return False
                return False
            # If a configured model is present (or a base match), consider Ollama available
            resolved = resolve_ollama_model_name()
            if resolved and any(resolved == m or resolved.split(":", 1)[0] == m.split(":", 1)[0] for m in models):
                return True
            return False
        except Exception:
            return False

    def generate(self, prompt: str) -> str:
        model_name = resolve_ollama_model_name()
        try:
            url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
            # Try common payload shapes used by different Ollama releases.
            payloads = [
                {"model": model_name, "input": prompt, "stream": False},
                {"model": model_name, "prompt": prompt, "stream": False},
                {"model": model_name, "text": prompt, "stream": False},
            ]
            last_exc: Exception | None = None
            for payload in payloads:
                try:
                    response = requests.post(url, json=payload, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    # Common response shapes: {"response": "..."} or {"result": {"output": "..."}} or list
                    if isinstance(data, dict):
                        if "response" in data:
                            return str(data.get("response", "")).strip()
                        if "result" in data and isinstance(data["result"], dict):
                            out = data["result"].get("output") or data["result"].get("response")
                            if out:
                                return str(out).strip()
                        # Some versions return {"outputs": [{"content": "..."}]}
                        outputs = data.get("outputs")
                        if outputs and isinstance(outputs, list) and outputs:
                            first = outputs[0]
                            if isinstance(first, dict):
                                txt = first.get("content") or first.get("output") or first.get("response")
                                if txt:
                                    return str(txt).strip()
                    elif isinstance(data, list) and data:
                        # maybe a list of message dicts
                        first = data[0]
                        if isinstance(first, dict):
                            txt = first.get("response") or first.get("output") or first.get("content")
                            if txt:
                                return str(txt).strip()
                    # Return the raw text body if the response shape is unexpected.
                    text = response.text.strip()
                    if text:
                        return text
                except Exception as exc:
                    last_exc = exc
                    continue
            runtime_error = _chat_model_runtime_error()
            if runtime_error:
                raise RuntimeError(runtime_error) from last_exc
            raise RuntimeError(
                f"Unable to get a response from Ollama model {config.OLLAMA_MODEL} at {config.OLLAMA_BASE_URL}"
            ) from last_exc
        except requests.RequestException as exc:
            runtime_error = _chat_model_runtime_error()
            if runtime_error:
                raise RuntimeError(runtime_error) from exc
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

    models = _ollama_model_catalog()
    if not models:
        return f"Unable to connect to Ollama at {base_url}"

    if not _ollama_model_available():
        return f"Ollama model {config.OLLAMA_MODEL} is not available. Run: ollama pull {_ollama_pull_model_name()}"

    return None


def _create_ollama_chat_model() -> Any:
    runtime_error = _chat_model_runtime_error()
    if runtime_error:
        raise RuntimeError(runtime_error)

    from langchain_ollama import ChatOllama

    model_name = resolve_ollama_model_name()
    try:
        return ChatOllama(model=model_name, temperature=0, base_url=config.OLLAMA_BASE_URL)
    except TypeError:
        try:
            return ChatOllama(model=model_name, base_url=config.OLLAMA_BASE_URL)
        except TypeError:
            try:
                return ChatOllama(model=model_name, temperature=0)
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

    return None


def current_runtime_status() -> dict[str, Any]:
    runtime_error = _chat_model_runtime_error()
    llm_available = runtime_error is None
    return {
        "provider": current_provider_name(),
        "model": current_model_name(),
        "llm_used": llm_available,
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "ollama_reachable": ollama_reachable(),
        "runtime_error": runtime_error,
    }
