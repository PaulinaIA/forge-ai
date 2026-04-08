"""Project path helpers.

Keep file system paths in one place so Streamlit/Docker/cwd differences don't
break RAG persistence or doc ingestion.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return repository root (the directory containing `src/`)."""
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = project_root()
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"


def ensure_data_dirs() -> None:
    """Create `data/` subdirs if missing (safe to call repeatedly)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

