"""
Dagster Software-Defined Assets Pipeline (V-22)

Three-asset pipeline:
  raw_data -> feature_engineered_data -> model_evaluation

Loads the Iris dataset, engineers features, trains a model, evaluates it,
and logs results to MLflow.
"""

import os
import logging

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

from dagster import (
    AssetExecutionContext,
    Definitions,
    MaterializeResult,
    MetadataValue,
    ScheduleDefinition,
    asset,
    define_asset_job,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

@asset(
    description="Load the Iris dataset as a pandas DataFrame with named columns.",
    metadata={"source": "sklearn.datasets", "dataset": "iris"},
)
def raw_data(context: AssetExecutionContext) -> pd.DataFrame:
    """Load iris data and return a DataFrame with feature + target columns."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    context.log.info("Loaded raw_data: %d rows, %d cols", *df.shape)
    return df


@asset(
    description=(
        "Engineer additional features (petal_ratio, sepal_ratio) and "
        "normalize numeric columns."
    ),
    metadata={"transformations": "ratio features + min-max normalization"},
)
def feature_engineered_data(
    context: AssetExecutionContext, raw_data: pd.DataFrame
) -> pd.DataFrame:
    """Add computed features and normalize."""
    df = raw_data.copy()

    # Computed features (guard against division by zero)
    df["petal_ratio"] = df["petal length (cm)"] / df["petal width (cm)"].replace(0, np.nan)
    df["sepal_ratio"] = df["sepal length (cm)"] / df["sepal width (cm)"].replace(0, np.nan)
    df["petal_ratio"] = df["petal_ratio"].fillna(0.0)
    df["sepal_ratio"] = df["sepal_ratio"].fillna(0.0)

    # Min-max normalization on feature columns (exclude target)
    feature_cols = [c for c in df.columns if c != "target"]
    for col in feature_cols:
        col_min, col_max = df[col].min(), df[col].max()
        if col_max - col_min > 0:
            df[col] = (df[col] - col_min) / (col_max - col_min)

    context.log.info(
        "feature_engineered_data: %d rows, %d cols (added petal_ratio, sepal_ratio)",
        *df.shape,
    )
    return df


@asset(
    description=(
        "Train a RandomForestClassifier, evaluate it, log to MLflow, "
        "and return a metrics dictionary."
    ),
    metadata={"model": "RandomForestClassifier", "tracking": "MLflow"},
)
def model_evaluation(
    context: AssetExecutionContext, feature_engineered_data: pd.DataFrame
) -> MaterializeResult:
    """Train, evaluate, and log to MLflow."""
    df = feature_engineered_data.copy()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

    # MLflow logging
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("dagster-iris-pipeline")

    with mlflow.start_run():
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(clf, artifact_path="model")

    context.log.info("Model metrics: %s", metrics)

    return MaterializeResult(
        metadata={
            "accuracy": MetadataValue.float(metrics["accuracy"]),
            "f1_score": MetadataValue.float(metrics["f1_score"]),
            "precision": MetadataValue.float(metrics["precision"]),
            "recall": MetadataValue.float(metrics["recall"]),
            "num_train_samples": MetadataValue.int(len(X_train)),
            "num_test_samples": MetadataValue.int(len(X_test)),
        }
    )


# ---------------------------------------------------------------------------
# Job & Schedule
# ---------------------------------------------------------------------------

iris_pipeline_job = define_asset_job(
    name="iris_pipeline_job",
    selection=[raw_data, feature_engineered_data, model_evaluation],
    description="Materialize all iris pipeline assets in sequence.",
)

# Daily schedule (commented out for PoC -- uncomment when ready)
# iris_daily_schedule = ScheduleDefinition(
#     job=iris_pipeline_job,
#     cron_schedule="0 0 * * *",  # midnight daily
#     name="iris_daily_schedule",
#     description="Run the iris pipeline every day at midnight.",
# )

# ---------------------------------------------------------------------------
# Dagster Definitions (entry point for dagster dev / workspace.yaml)
# ---------------------------------------------------------------------------

defs = Definitions(
    assets=[raw_data, feature_engineered_data, model_evaluation],
    jobs=[iris_pipeline_job],
    # schedules=[iris_daily_schedule],  # uncomment when ready
)


# ---------------------------------------------------------------------------
# Direct execution for local testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dagster import materialize

    logger.info("Materializing assets directly for testing...")
    result = materialize(
        assets=[raw_data, feature_engineered_data, model_evaluation],
    )
    if result.success:
        logger.info("All assets materialized successfully.")
    else:
        logger.error("Asset materialization failed.")
        raise SystemExit(1)
