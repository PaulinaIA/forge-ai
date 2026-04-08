"""Tool for tracking ML experiments using MLflow."""

import os
import json
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TrackExperimentInput(BaseModel):
    """Input model for the track_experiment tool."""
    experiment_name: str = Field(description="Name of the experiment (e.g., 'Churn Prediction').")
    run_name: str = Field(description="Name of this specific run (e.g., 'rf_baseline').")
    parameters: dict[str, Any] = Field(description="Dictionary of hyperparameters used.")
    metrics: dict[str, float] = Field(description="Dictionary of evaluation metrics.")


@tool("track_experiment", args_schema=TrackExperimentInput)
def track_experiment(experiment_name: str, run_name: str, parameters: dict[str, Any], metrics: dict[str, float]) -> str:
    """Log an ML experiment (hyperparameters and metrics) to MLflow.
    Use this tool whenever the user reports training a model, asks to save results,
    or wants to track different model configurations.
    """
    try:
        import mlflow
    except ImportError:
        return (
            "Error: MLflow is not installed. Please install it with "
            "`pip install mlflow` to track experiments."
        )

    # Set tracking URI if available in environment
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        # Set or create the experiment
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=run_name) as run:
            # Clean up parameters (convert complex types to strings if necessary)
            clean_params = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                            for k, v in parameters.items()}
            
            mlflow.log_params(clean_params)
            mlflow.log_metrics(metrics)
            
            run_id = run.info.run_id

        return (
            f"✅ Successfully logged experiment '{experiment_name}' (Run: '{run_name}') to MLflow.\n"
            f"- Run ID: {run_id}\n"
            f"- Logged {len(parameters)} parameters and {len(metrics)} metrics."
        )
    except Exception as e:
        return f"❌ Failed to log experiment to MLflow: {str(e)}"
