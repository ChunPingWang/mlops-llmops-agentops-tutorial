"""
MLflow Experiment Tracking & Model Registry Demo (V-20, V-21)

Trains a RandomForestClassifier on the Iris dataset, logs parameters,
metrics, and the model artifact to MLflow, then registers and promotes
the model through Staging to Production.
"""

import os
import logging

from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "poc-sklearn-demo"
REGISTERED_MODEL_NAME = "iris-classifier"

# --- hyper-parameters ---
PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
}


def train_and_log() -> str:
    """Train model, log everything to MLflow, and return the run ID."""

    mlflow.set_tracking_uri(TRACKING_URI)
    logger.info("MLflow tracking URI: %s", TRACKING_URI)

    # Create or reuse experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        logger.info("Created experiment '%s' (id=%s)", EXPERIMENT_NAME, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Using existing experiment '%s' (id=%s)", EXPERIMENT_NAME, experiment_id)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=PARAMS["random_state"]
    )

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("Started run %s", run_id)

        # --- log parameters ---
        mlflow.log_params(PARAMS)

        # --- train ---
        clf = RandomForestClassifier(**PARAMS)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # --- log metrics ---
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
        }
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            logger.info("  %s = %.4f", name, value)

        # --- log tags ---
        mlflow.set_tags({"model_type": "RandomForestClassifier", "dataset": "iris"})

        # --- log model artifact ---
        mlflow.sklearn.log_model(clf, artifact_path="model")
        model_uri = f"runs:/{run_id}/model"
        logger.info("Model URI: %s", model_uri)

    return run_id


def register_and_promote(run_id: str) -> None:
    """Register the model from *run_id* and transition it to Production."""

    client = MlflowClient(tracking_uri=TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"

    # Register (creates the registered model if it doesn't exist)
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    version = mv.version
    logger.info(
        "Registered model '%s' version %s (source: %s)",
        REGISTERED_MODEL_NAME,
        version,
        mv.source,
    )

    # Transition: None -> Staging
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Staging",
    )
    logger.info("Transitioned v%s to Staging", version)

    # Transition: Staging -> Production
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
    )
    logger.info("Transitioned v%s to Production", version)

    # Print summary
    latest = client.get_model_version(REGISTERED_MODEL_NAME, version)
    logger.info(
        "Registry info  name=%s  version=%s  stage=%s  status=%s",
        latest.name,
        latest.version,
        latest.current_stage,
        latest.status,
    )


def main() -> None:
    """Entry point: train, log, register, and promote."""
    try:
        run_id = train_and_log()
        register_and_promote(run_id)
        logger.info("Done.")
    except Exception:
        logger.exception("MLflow experiment failed")
        raise


if __name__ == "__main__":
    main()
