"""Code Generator tool — ML pipeline boilerplate generation."""

from __future__ import annotations

from langchain_core.tools import tool

PIPELINE_TEMPLATES = {
    "classification": {
        "description": "Binary/multiclass classification pipeline",
        "template": '''\
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import {model_class}
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import pandas as pd

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv("{file_path}")
X = df.drop(columns=["{target}"])
y = df["{target}"]

# ── Define column groups ─────────────────────────────────────
numeric_features = {numeric_features}
categorical_features = {categorical_features}

# ── Preprocessing ────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ── Full pipeline ────────────────────────────────────────────
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", {model_class}({model_params})),
])

# ── Evaluate ─────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="{scoring}")
print(f"{{scores.mean():.3f}} ± {{scores.std():.3f}} ({scoring})")

# ── Train final model ────────────────────────────────────────
pipeline.fit(X, y)
print(classification_report(y, pipeline.predict(X)))
''',
    },
    "regression": {
        "description": "Regression pipeline",
        "template": '''\
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import {model_class}
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# ── Load data ────────────────────────────────────────────────
df = pd.read_csv("{file_path}")
X = df.drop(columns=["{target}"])
y = df["{target}"]

# ── Define column groups ─────────────────────────────────────
numeric_features = {numeric_features}
categorical_features = {categorical_features}

# ── Preprocessing ────────────────────────────────────────────
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ── Full pipeline ────────────────────────────────────────────
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", {model_class}({model_params})),
])

# ── Evaluate ─────────────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error")
r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
print(f"RMSE: {{rmse_scores.mean():.3f}} ± {{rmse_scores.std():.3f}}")
print(f"R²: {{r2_scores.mean():.3f}} ± {{r2_scores.std():.3f}}")
''',
    },
}


@tool
def generate_code(
    task_type: str,
    target_column: str = "target",
    numeric_features: str = "[]",
    categorical_features: str = "[]",
    model_preference: str = "auto",
    file_path: str = "data.csv",
) -> str:
    """Generate a scikit-learn ML pipeline with preprocessing, training, and evaluation."""
    task_type = task_type.lower().strip()

    if task_type not in PIPELINE_TEMPLATES:
        return f"Unknown task type '{task_type}'. Supported: {', '.join(PIPELINE_TEMPLATES.keys())}"

    model_class, model_params, scoring = _select_model(task_type, model_preference)
    template = PIPELINE_TEMPLATES[task_type]["template"]

    code = template.format(
        file_path=file_path,
        target=target_column,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_class=model_class,
        model_params=model_params,
        scoring=scoring,
    )

    return f"```python\n{code}\n```"


def _select_model(task_type: str, preference: str) -> tuple[str, str, str]:
    models = {
        "classification": {
            "auto": (
                "GradientBoostingClassifier",
                "n_estimators=200, random_state=42",
                "f1_weighted",
            ),
            "gradient_boosting": (
                "GradientBoostingClassifier",
                "n_estimators=200, learning_rate=0.1, random_state=42",
                "f1_weighted",
            ),
            "random_forest": (
                "RandomForestClassifier",
                "n_estimators=200, random_state=42, n_jobs=-1",
                "f1_weighted",
            ),
            "linear": ("LogisticRegression", "max_iter=1000, random_state=42", "f1_weighted"),
        },
        "regression": {
            "auto": ("GradientBoostingRegressor", "n_estimators=200, random_state=42", "r2"),
            "gradient_boosting": (
                "GradientBoostingRegressor",
                "n_estimators=200, learning_rate=0.1, random_state=42",
                "r2",
            ),
            "random_forest": (
                "RandomForestRegressor",
                "n_estimators=200, random_state=42, n_jobs=-1",
                "r2",
            ),
            "linear": ("Ridge", "alpha=1.0", "r2"),
        },
    }

    pref = preference.lower().strip()
    if pref not in models.get(task_type, {}):
        pref = "auto"

    return models[task_type][pref]

