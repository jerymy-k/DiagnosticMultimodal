import os

import mlflow

from src.utils.config import PROJECT_ROOT

TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").as_uri()


def setup_mlflow(experiment: str) -> None:
    os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(experiment)
