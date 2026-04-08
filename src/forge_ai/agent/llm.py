"""LLM client — OpenAI-compatible SDK supporting Groq, Gemini, and OpenAI.

This module holds provider defaults and a thin client wrapper. The LangChain
agent uses `langchain_openai.ChatOpenAI` directly, but keeping this logic here
centralizes provider config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

# ─── Provider configurations ─────────────────────────────────────────────────

PROVIDERS: dict[str, dict[str, str | None]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "default_embedding_model": None,  # No embeddings — use HuggingFace
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
        "default_embedding_model": "text-embedding-004",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
        "default_embedding_model": "text-embedding-3-small",
    },
}


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    provider: str = "groq"
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider '{self.provider}'. Available: {', '.join(PROVIDERS.keys())}"
            )
        if self.model is None:
            self.model = str(PROVIDERS[self.provider]["default_model"])


def get_llm_client(config: LLMConfig | None = None) -> OpenAI:
    """Create an OpenAI-compatible client for the configured provider."""
    if config is None:
        config = LLMConfig()

    provider_cfg = PROVIDERS[config.provider]
    api_key = os.getenv(str(provider_cfg["env_key"]))

    if not api_key:
        raise ValueError(
            f"API key not found for {config.provider}. "
            f"Set {provider_cfg['env_key']} in your .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url=str(provider_cfg["base_url"]),
    )


def chat_completion(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """Simple chat completion wrapper for the OpenAI-compatible client."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""

