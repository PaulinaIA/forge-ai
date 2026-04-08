"""Dataset Analyzer tool — CSV/DataFrame profiling and EDA."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from forge_ai.io import read_csv_smart

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


@tool
def analyze_dataset(file_path: str) -> str:
    """Profile a CSV dataset: shape, types, missing values, distributions, and detected issues."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found at '{file_path}'"

    try:
        df = read_csv_smart(path)
    except Exception as e:
        return (
            "Error reading CSV. This often happens when the file uses a different delimiter "
            "(e.g. ';' instead of ',') or a non-UTF8 encoding.\n\n"
            f"Details: {e}"
        )

    profile = _build_profile(df)
    return _format_profile(profile, file_path)


def _build_profile(df: pd.DataFrame) -> dict:
    import numpy as np

    n_rows, n_cols = df.shape
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].sort_values(ascending=False)
    total_missing = missing.sum()
    missing_pct = (total_missing / (n_rows * n_cols)) * 100 if n_rows and n_cols else 0.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    issues: list[str] = []

    for col in df.columns:
        if df[col].nunique() <= 1:
            issues.append(f"'{col}' is constant (1 unique value) — drop it")
        elif df[col].nunique() / max(n_rows, 1) > 0.95 and col in categorical_cols:
            issues.append(
                f"'{col}' has very high cardinality ({df[col].nunique()} unique) — possible ID column"
            )

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [
            (col, row, corr_matrix.loc[row, col])
            for col in upper_tri.columns
            for row in upper_tri.index
            if upper_tri.loc[row, col] > 0.9
        ]
        for col1, col2, corr_val in high_corr[:3]:
            issues.append(
                f"'{col1}' and '{col2}' are highly correlated ({corr_val:.2f}) — consider dropping one"
            )

    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 2:
            direction = "right" if skewness > 0 else "left"
            issues.append(f"'{col}' is heavily {direction}-skewed ({skewness:.1f}) — consider log transform")

    potential_targets: list[tuple[str, str]] = []
    for col in df.columns:
        nunique = df[col].nunique()
        if nunique == 2:
            potential_targets.append((col, "binary classification"))
        elif 3 <= nunique <= 20 and col in categorical_cols:
            potential_targets.append((col, "multiclass classification"))
    if numeric_cols:
        potential_targets.append((numeric_cols[-1], "regression"))

    numeric_stats: dict[str, dict[str, float | int]] = {}
    for col in numeric_cols[:10]:
        numeric_stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "std": round(float(df[col].std()), 2),
            "min": round(float(df[col].min()), 2),
            "max": round(float(df[col].max()), 2),
            "nulls": int(df[col].isnull().sum()),
        }

    return {
        "shape": (n_rows, n_cols),
        "memory_mb": round(float(memory_mb), 2),
        "columns": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
            "boolean": bool_cols,
        },
        "missing": {
            "total": int(total_missing),
            "pct": round(float(missing_pct), 1),
            "by_column": {col: int(val) for col, val in missing_cols.items()},
        },
        "numeric_stats": numeric_stats,
        "issues": issues,
        "potential_targets": potential_targets[:3],
    }


def _format_profile(profile: dict, file_path: str) -> str:
    n_rows, n_cols = profile["shape"]
    cols = profile["columns"]

    lines = [
        f"📊 **Dataset Profile: {Path(file_path).name}**\n",
        f"• Shape: {n_rows:,} rows × {n_cols} columns ({profile['memory_mb']} MB)",
        f"• Numeric: {len(cols['numeric'])} | Categorical: {len(cols['categorical'])} | "
        f"Datetime: {len(cols['datetime'])} | Boolean: {len(cols['boolean'])}",
    ]

    m = profile["missing"]
    if m["total"] > 0:
        lines.append(f"\n⚠️ **Missing Values**: {m['total']:,} ({m['pct']}%)")
        for col, count in list(m["by_column"].items())[:5]:
            pct = round(count / max(n_rows, 1) * 100, 1)
            lines.append(f"  - '{col}': {count:,} ({pct}%)")
    else:
        lines.append("\n✅ No missing values")

    if profile["numeric_stats"]:
        lines.append("\n📈 **Numeric Summary** (first 10):")
        for col, stats in profile["numeric_stats"].items():
            lines.append(
                f"  - '{col}': mean={stats['mean']}, std={stats['std']}, "
                f"range=[{stats['min']}, {stats['max']}]"
            )

    if profile["issues"]:
        lines.append(f"\n🔍 **Detected Issues** ({len(profile['issues'])}):")
        for issue in profile["issues"]:
            lines.append(f"  - {issue}")

    if profile["potential_targets"]:
        lines.append("\n🎯 **Potential Target Columns**:")
        for col, task_type in profile["potential_targets"]:
            lines.append(f"  - '{col}' → {task_type}")

    return "\n".join(lines)

