from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if src.exists():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


def test_imports_smoke() -> None:
    _ensure_src_on_path()

    import forge_ai  # noqa: F401
    from forge_ai.agent.forge import ForgeAgent

    # Avoid requiring real credentials in CI/dev machines.
    os.environ.setdefault("GROQ_API_KEY", "test-key")

    agent = ForgeAgent(provider="groq", verbose=False)
    assert set(agent.tool_names) >= {
        "analyze_dataset",
        "rag_query",
        "generate_code",
        "evaluate_model",
        "suggest_preprocessing",
    }


def test_tools_registry_smoke() -> None:
    _ensure_src_on_path()

    from forge_ai.tools import get_all_tools

    tools = get_all_tools()
    assert len(tools) >= 5
    assert {t.name for t in tools} >= {
        "analyze_dataset",
        "rag_query",
        "generate_code",
        "evaluate_model",
        "suggest_preprocessing",
    }


def test_read_csv_smart_detects_semicolon_delimiter(tmp_path: Path) -> None:
    _ensure_src_on_path()

    from forge_ai.io import read_csv_smart

    p = tmp_path / "sample_semicolon.csv"
    p.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")

    df = read_csv_smart(p)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)

