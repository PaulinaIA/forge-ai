"""RAG Retriever — Semantic search over ML documentation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from forge_ai.paths import VECTORSTORE_DIR

if TYPE_CHECKING:  # pragma: no cover
    from langchain_community.vectorstores import Chroma

COLLECTION_NAME = "ml_docs"

_vectorstore: "Chroma | None" = None


def get_vectorstore(persist_dir: Path = VECTORSTORE_DIR) -> "Chroma":
    """Get or create the LangChain-wrapped vectorstore."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
        )
    return _vectorstore


@tool
def rag_query(query: str, category: str = "", top_k: int = 5) -> str:
    """Search the ML documentation knowledge base for relevant information."""
    try:
        vectorstore = get_vectorstore()
    except Exception:
        return (
            "RAG knowledge base not initialized. "
            "Run 'python scripts/ingest_docs.py' first. "
            "Answering from general knowledge instead."
        )

    search_kwargs: dict[str, object] = {"k": top_k}
    if category:
        search_kwargs["filter"] = {"category": category}

    docs = vectorstore.similarity_search(query, **search_kwargs)

    if not docs:
        return f"No relevant documentation found for: '{query}'. Answering from general knowledge."

    results: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        cat = doc.metadata.get("category", "general")
        results.append(f"[Source: {source} | Category: {cat}]\n{doc.page_content}")

    return "\n\n---\n\n".join(results)

