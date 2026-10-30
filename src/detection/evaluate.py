import logging
from pathlib import Path

import mlflow
import pandas as pd
from ultralytics import YOLO

from src.utils.config import load_config, project_path, read_json, staging_dir, write_json
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

METRIC_NAMES = ["precision", "recall", "mAP50", "mAP50-95"]


def metrics_table(metrics, class_names):
    box = metrics.box
    rows = {class_names[c]: box.class_result(i) for i, c in enumerate(box.ap_class_index)}
    rows["all"] = box.mean_results()
    return pd.DataFrame(rows, index=METRIC_NAMES).T


def evaluate_detector() -> dict:
    pipeline_config = load_config("pipeline")["detection"]
    train_args = load_config("detection")
    summary = read_json(staging_dir() / "detector.json")
    setup_mlflow(pipeline_config["experiment"])

    metrics = YOLO(staging_dir() / "detector.pt").val(
        data=str(project_path(train_args["data"])), split="test", imgsz=train_args["imgsz"],
        batch=train_args["batch"], project=str(project_path(pipeline_config["runs_dir"])),
        name=f"{summary['config']}_test", exist_ok=True,
    )
    table = metrics_table(metrics, summary["class_names"])
    test_metrics = {name: float(value) for name, value in table.loc["all"].items()}
    logger.info("Détection - test :\n%s", table.round(3).to_string())

    with mlflow.start_run(run_id=summary["run_id"]):
        mlflow.log_metrics({f"test/{name}": value for name, value in test_metrics.items()})
        mlflow.log_text(table.round(4).to_string(), "test/metrics_by_class.txt")
        mlflow.log_artifacts(str(Path(metrics.save_dir)), artifact_path="test")

    summary["test_metrics"] = test_metrics
    summary["test_dir"] = str(metrics.save_dir)
    write_json(staging_dir() / "detector.json", summary)
    return test_metrics
