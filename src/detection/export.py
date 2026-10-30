import logging
import shutil
from datetime import date
from pathlib import Path

import mlflow
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from src.utils.config import load_config, project_path, read_json, staging_dir, write_json
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

REGISTERED_NAME = "brain-tumor-detector"
SELECTION_METRIC = "mAP50-95"
PYFUNC_MODEL = Path(__file__).with_name("pyfunc_model.py")
OUTPUT_COLUMNS = {"image_path": "string", "class": "string", "confidence": "double",
                  "x1": "double", "y1": "double", "x2": "double", "y2": "double"}


def current_score(meta_path):
    if not meta_path.exists():
        return None
    return read_json(meta_path)["test_metrics"].get(SELECTION_METRIC)


def copy_artifacts(summary, artifacts_dir):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = project_path(load_config("detection")["data"])
    shutil.copy2(data_yaml, artifacts_dir / "data.yaml")
    shutil.copy2(Path(summary["run_dir"]) / "results.png", artifacts_dir / "results.png")
    for plot_name in ["BoxPR_curve.png", "confusion_matrix_normalized.png"]:
        shutil.copy2(Path(summary["test_dir"]) / plot_name, artifacts_dir / f"test_{plot_name}")


def save_detector(force: bool = False) -> dict:
    pipeline_config = load_config("pipeline")
    models_dir = project_path(pipeline_config["models_dir"])
    summary = read_json(staging_dir() / "detector.json")
    new_score = summary["test_metrics"][SELECTION_METRIC]
    old_score = current_score(models_dir / "detector_meta.json")

    if old_score is not None and new_score <= old_score and not force:
        logger.info("Détecteur non promu : %s %.4f <= modèle actuel %.4f", SELECTION_METRIC, new_score, old_score)
        return {"promoted": False, "score": new_score, "current": old_score}

    model_path = models_dir / "detector.pt"
    shutil.copy2(staging_dir() / "detector.pt", model_path)
    copy_artifacts(summary, models_dir / "detector")
    meta = {
        "class_names": summary["class_names"],
        "img_size": summary["img_size"],
        "config": summary["config"],
        "mlflow_run_id": summary["run_id"],
        "test_metrics": {k: round(v, 4) for k, v in summary["test_metrics"].items()},
    }
    write_json(models_dir / "detector_meta.json", meta)

    setup_mlflow(pipeline_config["detection"]["experiment"])
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "image_path")]),
        outputs=Schema([ColSpec(dtype, name) for name, dtype in OUTPUT_COLUMNS.items()]),
    )
    with mlflow.start_run(run_id=summary["run_id"]):
        model_info = mlflow.pyfunc.log_model(
            name="detector", python_model=str(PYFUNC_MODEL), artifacts={"weights": str(model_path)},
            signature=signature,
        )
        mlflow.log_artifact(str(models_dir / "detector_meta.json"))
        mlflow.log_artifact(str(models_dir / "detector" / "data.yaml"))

    client = mlflow.MlflowClient()
    version = mlflow.register_model(model_info.model_uri, REGISTERED_NAME)
    test = meta["test_metrics"]
    client.update_model_version(
        REGISTERED_NAME, version.version,
        description=(f"Run {summary['run_id']} ({summary['config']}) - test mAP50 {test['mAP50']}, "
                     f"mAP50-95 {test['mAP50-95']}, recall {test['recall']} - enregistré le {date.today()}"),
    )
    for key, value in {"task": "detection", "framework": "ultralytics", "config": summary["config"]}.items():
        client.set_model_version_tag(REGISTERED_NAME, version.version, key, value)
    client.set_registered_model_alias(REGISTERED_NAME, "champion", version.version)
    logger.info("Détecteur promu : %s version %s (champion)", REGISTERED_NAME, version.version)
    return {"promoted": True, "score": new_score, "version": version.version}
