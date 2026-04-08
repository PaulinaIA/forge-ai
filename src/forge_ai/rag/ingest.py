"""RAG ingestion — Chunk and embed ML documentation into ChromaDB."""

from __future__ import annotations

from pathlib import Path

from forge_ai.paths import DOCS_DIR, VECTORSTORE_DIR, ensure_data_dirs

COLLECTION_NAME = "ml_docs"


def ingest_docs(
    docs_dir: Path = DOCS_DIR,
    persist_dir: Path = VECTORSTORE_DIR,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> int:
    """Ingest ML documentation into ChromaDB with HuggingFace embeddings."""
    import chromadb
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings

    ensure_data_dirs()
    documents: list[str] = []
    metadatas: list[dict] = []

    for filepath in sorted(docs_dir.glob("**/*.md")):
        content = filepath.read_text(encoding="utf-8")
        source = filepath.relative_to(docs_dir).as_posix()
        category = _detect_category(filepath.name)

        documents.append(content)
        metadatas.append({"source": source, "category": category, "filename": filepath.name})

    if not documents:
        raise FileNotFoundError(
            f"No markdown files found in {docs_dir}. "
            "Add docs under data/docs or run a download script first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )

    chunks: list[str] = []
    chunk_metadatas: list[dict] = []
    for doc, meta in zip(documents, metadatas):
        splits = splitter.split_text(doc)
        for i, chunk in enumerate(splits):
            chunks.append(chunk)
            chunk_metadatas.append({**meta, "chunk_index": i})

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

    print(f"Generating embeddings for {len(chunks)} chunks...")
    vectors = embeddings.embed_documents(chunks)

    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        client.delete_collection(collection_name)
    except ValueError:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "ML/DL documentation for RAG"},
    )

    collection.add(
        documents=chunks,
        embeddings=vectors,
        metadatas=chunk_metadatas,
        ids=[f"doc-{i:05d}" for i in range(len(chunks))],
    )

    print(f"✓ Ingested {len(chunks)} chunks from {len(documents)} documents")
    return len(chunks)


def _detect_category(filename: str) -> str:
    name = filename.lower()
    if "sklearn" in name or "scikit" in name:
        return "scikit-learn"
    if "pytorch" in name or "torch" in name:
        return "pytorch"
    if "pandas" in name:
        return "pandas"
    if "preprocessing" in name or "feature" in name:
        return "preprocessing"
    if "evaluation" in name or "metrics" in name:
        return "evaluation"
    if "best_practice" in name or "patterns" in name:
        return "best-practices"
    return "general"


if __name__ == "__main__":
    count = ingest_docs()
    print(f"Done: {count} chunks indexed.")

