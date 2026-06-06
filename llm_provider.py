"""Provider abstraction for optional LLM-backed answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests

import config


class LLMProvider(Protocol):
    name: str

    def generate(self, prompt: str) -> str:
        ...


@dataclass
class OllamaProvider:
    name: str = "ollama"

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
                json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama is configured but not reachable. Start it with `ollama serve` "
                f"and make sure model `{config.OLLAMA_MODEL}` is pulled. Error: {exc}"
            ) from exc


@dataclass
class OpenAIProvider:
    name: str = "openai"

    def generate(self, prompt: str) -> str:
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=config.OPENAI_API_KEY)
        return llm.invoke(prompt).content


@dataclass
class AzureOpenAIProvider:
    name: str = "azure"

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
            temperature=0,
        )
        return llm.invoke(prompt).content


def get_llm_provider() -> LLMProvider:
    if config.LLM_PROVIDER == "openai":
        return OpenAIProvider()
    if config.LLM_PROVIDER in {"azure", "azure_openai"}:
        return AzureOpenAIProvider()
    return OllamaProvider()
