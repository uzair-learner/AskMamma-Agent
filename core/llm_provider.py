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
                    # Fallback: return text body
                    text = response.text.strip()
                    if text:
                        return text
                except Exception as exc:
                    last_exc = exc
                    continue
            # If none of the payloads worked, raise a descriptive error
            raise RuntimeError(
                "Ollama is configured but not reachable or the model is not available. Start it with `ollama serve` and make sure model '{model_name}' is pulled."
            ) from last_exc
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


def _dependency_installed(module_name: str) -> bool:
    return find_spec(module_name) is not None


def llm_runtime_available() -> bool:
    if config.LLM_PROVIDER == "openai":
        return bool(config.OPENAI_API_KEY and _dependency_installed("langchain_openai"))

    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        return bool(
            AzureOpenAIProvider().available()
            and _dependency_installed("langchain_openai")
        )

    return bool(
        OllamaProvider().available()
        and _dependency_installed("langchain_ollama")
    )


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
            # Try common LangChain Ollama chat class names and constructor signatures.
            try:
                from langchain_ollama import ChatOllama as _ChatClass
            except Exception:
                try:
                    from langchain_ollama import Ollama as _ChatClass
                except Exception:
                    return None

            model_name = resolve_ollama_model_name()
            # Attempt several constructor signatures for compatibility.
            try:
                return _ChatClass(model=model_name, temperature=0, base_url=config.OLLAMA_BASE_URL)
            except TypeError:
                try:
                    return _ChatClass(model=model_name, base_url=config.OLLAMA_BASE_URL)
                except TypeError:
                    try:
                        return _ChatClass(model=model_name, temperature=0)
                    except Exception:
                        return None
        except ImportError:
            return None

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
    llm_available = llm_runtime_available()
    return {
        "provider": current_provider_name(),
        "model": current_model_name(),
        "llm_used": llm_available,
        "ollama_base_url": config.OLLAMA_BASE_URL,
        "ollama_reachable": ollama_reachable(),
    }
