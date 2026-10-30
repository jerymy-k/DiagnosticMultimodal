import logging
import shutil
from datetime import date

import mlflow
import mlflow.keras

from src.classification.model import tf
from src.utils.config import load_config, project_path, read_json, staging_dir, write_json
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

REGISTERED_NAME = "brain-tumor-classifier"
SELECTION_METRIC = "test_f1_macro"


def current_score(meta_path):
    if not meta_path.exists():
        return None
    return read_json(meta_path)["test_metrics"].get(SELECTION_METRIC)


def save_classifier(force: bool = False) -> dict:
    config = load_config("classification")
    models_dir = project_path(load_config("pipeline")["models_dir"])
    summary = read_json(staging_dir() / "classifier.json")
    new_score = summary["test_metrics"][SELECTION_METRIC]
    old_score = current_score(models_dir / "classifier_meta.json")

    if old_score is not None and new_score <= old_score and not force:
        logger.info("Classifieur non promu : %s %.4f <= modèle actuel %.4f", SELECTION_METRIC, new_score, old_score)
        return {"promoted": False, "score": new_score, "current": old_score}

    model_path = models_dir / "classifier.keras"
    shutil.copy2(staging_dir() / "classifier.keras", model_path)
    meta = {
        "class_names": summary["class_names"],
        "img_size": summary["img_size"],
        "input": "RGB, pixels bruts 0-255 (normalisation incluse dans le modèle)",
        "config": summary["config"],
        "mlflow_run_id": summary["run_id"],
        "test_metrics": {k: round(v, 4) for k, v in summary["test_metrics"].items()},
    }
    write_json(models_dir / "classifier_meta.json", meta)

    setup_mlflow(config["experiment"])
    with mlflow.start_run(run_id=summary["run_id"]):
        model_info = mlflow.keras.log_model(tf.keras.models.load_model(model_path), name="classifier")
        mlflow.log_artifact(str(models_dir / "classifier_meta.json"))

    client = mlflow.MlflowClient()
    version = mlflow.register_model(model_info.model_uri, REGISTERED_NAME)
    test = meta["test_metrics"]
    client.update_model_version(
        REGISTERED_NAME, version.version,
        description=(f"Run {summary['run_id']} ({summary['config']}) - test accuracy {test['test_accuracy']}, "
                     f"F1 macro {test['test_f1_macro']}, recall macro {test['test_recall_macro']} - "
                     f"enregistré le {date.today()}"),
    )
    for key, value in {"task": "classification", "framework": "keras", "config": summary["config"]}.items():
        client.set_model_version_tag(REGISTERED_NAME, version.version, key, value)
    client.set_registered_model_alias(REGISTERED_NAME, "champion", version.version)
    logger.info("Classifieur promu : %s version %s (champion)", REGISTERED_NAME, version.version)
    return {"promoted": True, "score": new_score, "version": version.version}
