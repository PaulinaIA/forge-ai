"""Ingest markdown docs into the local Chroma vectorstore.

Usage:
  python scripts/ingest_docs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without `pip install -e .`
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from forge_ai.rag.ingest import ingest_docs  # noqa: E402


def main() -> None:
    count = ingest_docs()
    print(f"Done: {count} chunks indexed.")


if __name__ == "__main__":
    main()

