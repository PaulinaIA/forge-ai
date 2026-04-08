"""LangChain tools exposed to the agent.

Important: keep this module light on import. Heavy dependencies (pandas, sklearn,
chromadb, sentence-transformers) should be imported inside tool functions, not
at module import time.
"""

from __future__ import annotations

__all__ = ["get_all_tools"]


def get_all_tools():
    """Return the full list of tools available to the agent."""
    from forge_ai.rag.retriever import rag_query
    from forge_ai.tools.code_generator import generate_code
    from forge_ai.tools.dataset_analyzer import analyze_dataset
    from forge_ai.tools.model_evaluator import evaluate_model
    from forge_ai.tools.preprocessing import suggest_preprocessing
    from forge_ai.tools.experiment_tracker import track_experiment

    return [analyze_dataset, rag_query, generate_code, evaluate_model, suggest_preprocessing, track_experiment]


def __getattr__(name: str):
    """Lazy-load tool symbols for convenience imports."""
    if name == "rag_query":
        from forge_ai.rag.retriever import rag_query

        return rag_query
    if name == "generate_code":
        from forge_ai.tools.code_generator import generate_code

        return generate_code
    if name == "analyze_dataset":
        from forge_ai.tools.dataset_analyzer import analyze_dataset

        return analyze_dataset
    if name == "evaluate_model":
        from forge_ai.tools.model_evaluator import evaluate_model

        return evaluate_model
    if name == "suggest_preprocessing":
        from forge_ai.tools.preprocessing import suggest_preprocessing

        return suggest_preprocessing
    if name == "track_experiment":
        from forge_ai.tools.experiment_tracker import track_experiment

        return track_experiment
    if name == "get_all_tools":
        return get_all_tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        {
            "analyze_dataset",
            "evaluate_model",
            "generate_code",
            "get_all_tools",
            "rag_query",
            "suggest_preprocessing",
            "track_experiment",
        }
    )

