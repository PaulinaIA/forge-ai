"""Preprocessing Advisor tool — Data cleaning & feature engineering suggestions."""

from __future__ import annotations

from pathlib import Path
from langchain_core.tools import tool

from forge_ai.io import read_csv_smart



@tool
def suggest_preprocessing(file_path: str, target_column: str = "") -> str:
    """Analyze a dataset and recommend specific preprocessing strategies."""
    import numpy as np
    import pandas as pd

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

    recommendations: list[str] = []

    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]

    if len(missing_cols) > 0:
        recommendations.append("## 🩹 Missing Value Strategy\n")
        for col in missing_cols.index:
            pct = missing_cols[col] / len(df) * 100
            dtype = df[col].dtype

            if pct > 60:
                recommendations.append(
                    f"- **`{col}`** ({pct:.0f}% missing): **Drop column** — "
                    "too many missing values to impute reliably"
                )
            elif pct > 30:
                recommendations.append(
                    f"- **`{col}`** ({pct:.0f}% missing): **Add missingness indicator** + impute. "
                    "The pattern of missingness itself may be informative"
                )
            elif np.issubdtype(dtype, np.number):
                skew = float(df[col].skew())
                if abs(skew) > 1:
                    recommendations.append(
                        f"- **`{col}`** ({pct:.1f}% missing, skewed): **`SimpleImputer(strategy='median')`**"
                    )
                else:
                    recommendations.append(
                        f"- **`{col}`** ({pct:.1f}% missing, ~symmetric): **`SimpleImputer(strategy='mean')`**"
                    )
            else:
                recommendations.append(
                    f"- **`{col}`** ({pct:.1f}% missing, categorical): **`SimpleImputer(strategy='most_frequent')`**"
                )
    else:
        recommendations.append("## ✅ Missing Values: None detected\n")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        recommendations.append("\n## 🏷️ Encoding Strategy\n")
        for col in cat_cols:
            n_unique = int(df[col].nunique())
            if n_unique == 2:
                recommendations.append(f"- **`{col}`** (2 categories): **`LabelEncoder`** or map to 0/1")
            elif n_unique <= 10:
                recommendations.append(f"- **`{col}`** ({n_unique} categories): **`OneHotEncoder`**")
            elif n_unique <= 50:
                recommendations.append(
                    f"- **`{col}`** ({n_unique} categories): **`TargetEncoder`** "
                    "(high cardinality — OneHot would create too many columns)"
                )
            else:
                recommendations.append(
                    f"- **`{col}`** ({n_unique} categories): **Consider dropping or hashing** — "
                    "very high cardinality, may be an ID column"
                )

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        recommendations.append("\n## 📏 Scaling Strategy\n")
        has_outliers = False
        for col in num_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_pct = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).mean() * 100
            if outlier_pct > 5:
                has_outliers = True

        if has_outliers:
            recommendations.append(
                "- Outliers detected → **`RobustScaler`** (uses median/IQR, robust to outliers)"
            )
        else:
            recommendations.append("- No significant outliers → **`StandardScaler`** (z-score normalization)")

    recommendations.append("\n## ⚙️ Feature Engineering Suggestions\n")

    for col in df.columns:
        sample = df[col].dropna().head(5)
        try:
            pd.to_datetime(sample)
            recommendations.append(
                f"- **`{col}`** looks like a date → extract: year, month, day_of_week, is_weekend"
            )
        except (ValueError, TypeError):
            pass

    for col in num_cols:
        skew = abs(float(df[col].skew()))
        if skew > 2:
            recommendations.append(f"- **`{col}`** is heavily skewed ({skew:.1f}) → apply `np.log1p()` transform")

    if target_column and target_column in df.columns:
        target = df[target_column]
        if target.nunique() <= 10:
            counts = target.value_counts(normalize=True)
            minority_pct = float(counts.min() * 100)
            if minority_pct < 20:
                recommendations.append(
                    "\n## ⚖️ Class Imbalance\n"
                    f"- Minority class: {minority_pct:.1f}% → Consider:\n"
                    "  - `class_weight='balanced'` in the model\n"
                    "  - SMOTE oversampling (`imblearn.over_sampling.SMOTE`)\n"
                    "  - Stratified cross-validation (`StratifiedKFold`)"
                )

    return "\n".join(recommendations)

