"""Model Evaluator tool — Metrics computation and diagnostics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@tool
def evaluate_model(
    y_true: str,
    y_pred: str,
    task_type: str = "classification",
    y_proba: str = "[]",
) -> str:
    """Evaluate model predictions: compute metrics, detect issues, suggest improvements."""
    import numpy as np

    try:
        y_t = np.array(json.loads(y_true))
        y_p = np.array(json.loads(y_pred))
    except (json.JSONDecodeError, ValueError) as e:
        return f"Error parsing inputs: {e}. Provide JSON arrays like '[1, 0, 1]'."

    if len(y_t) != len(y_p):
        return f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})"

    tt = task_type.lower().strip()
    if tt == "classification":
        return _evaluate_classification(y_t, y_p, y_proba)
    if tt == "regression":
        return _evaluate_regression(y_t, y_p)
    return f"Unknown task type '{task_type}'. Use 'classification' or 'regression'."


def _evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba_str: str) -> str:
    import numpy as np

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    n_classes = len(np.unique(y_true))
    is_binary = n_classes == 2
    average = "binary" if is_binary else "weighted"

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    lines = [
        "📊 **Classification Evaluation**\n",
        f"• Accuracy:  {acc:.3f}",
        f"• Precision: {prec:.3f}",
        f"• Recall:    {rec:.3f}",
        f"• F1 Score:  {f1:.3f}",
    ]

    try:
        y_proba = np.array(json.loads(y_proba_str))
        if len(y_proba) == len(y_true) and is_binary:
            auc = roc_auc_score(y_true, y_proba)
            lines.append(f"• ROC-AUC:   {auc:.3f}")
    except (json.JSONDecodeError, ValueError):
        pass

    lines.append(f"\n📋 **Confusion Matrix**:\n{cm}")
    lines.append("\n🔍 **Diagnostics**:")

    # Class imbalance check
    try:
        class_counts = np.bincount(y_true.astype(int))
        if len(class_counts) >= 2:
            ratio = class_counts.min() / class_counts.max()
            if ratio < 0.3:
                lines.append(
                    f"  ⚠️ Class imbalance detected (ratio: {ratio:.2f}). "
                    "Consider: SMOTE, class_weight='balanced', or threshold tuning."
                )
    except Exception:
        # Non-integer labels etc.
        pass

    if is_binary:
        if prec > rec + 0.15:
            lines.append(
                "  ⚠️ Precision >> Recall: Model is conservative. "
                "Lower the decision threshold to catch more positives."
            )
        elif rec > prec + 0.15:
            lines.append(
                "  ⚠️ Recall >> Precision: Too many false positives. "
                "Raise the threshold or add more discriminative features."
            )

    if acc > 0.98:
        lines.append(
            "  ⚠️ Very high accuracy (>98%). Possible data leakage or overfitting. "
            "Verify with cross-validation."
        )

    return "\n".join(lines)


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    import numpy as np

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    )

    residuals = y_true - y_pred

    lines = [
        "📊 **Regression Evaluation**\n",
        f"• RMSE: {rmse:.4f}",
        f"• MAE:  {mae:.4f}",
        f"• R²:   {r2:.4f}",
        f"• MAPE: {mape:.1f}%",
        "\n📋 **Residual Stats**:",
        f"• Mean: {float(residuals.mean()):.4f} (should be ~0)",
        f"• Std:  {float(residuals.std()):.4f}",
        f"• Min:  {float(residuals.min()):.4f}",
        f"• Max:  {float(residuals.max()):.4f}",
    ]

    lines.append("\n🔍 **Diagnostics**:")

    if r2 < 0:
        lines.append("  ❌ Negative R²: Model is worse than predicting the mean. Check your features.")
    elif r2 < 0.3:
        lines.append(
            "  ⚠️ Low R² (<0.3): Model explains little variance. Consider more features or nonlinear models."
        )

    if abs(float(residuals.mean())) > float(residuals.std()) * 0.1:
        lines.append("  ⚠️ Residuals are biased (mean ≠ 0). Model may have systematic under/over-prediction.")

    if r2 > 0.99:
        lines.append("  ⚠️ R² > 0.99: Possible data leakage. Verify with proper train/test splits.")

    return "\n".join(lines)

