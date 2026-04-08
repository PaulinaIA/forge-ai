"""Agent orchestration and LLM integration."""

from __future__ import annotations

__all__ = ["ForgeAgent"]


def __getattr__(name: str):
    if name == "ForgeAgent":
        from forge_ai.agent.forge import ForgeAgent

        return ForgeAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

