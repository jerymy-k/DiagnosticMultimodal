import contextlib
import logging
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from ultralytics import YOLO, settings

from src.utils.config import load_config, project_path, staging_dir, write_json
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

SELECTION_COLUMN = "metrics/mAP50-95(B)"


def train_candidate(weights, train_args, runs_dir, quick):
    run_name = Path(weights).stem
    pretrained_dir = project_path("models/pretrained")
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(pretrained_dir):
        YOLO(weights).train(**train_args, project=str(runs_dir), name=run_name, exist_ok=True)
    run_id = mlflow.last_active_run().info.run_id
    mlflow.MlflowClient().set_tag(run_id, "source", "pipeline")
    mlflow.MlflowClient().set_tag(run_id, "quick", str(quick))
    results = pd.read_csv(runs_dir / run_name / "results.csv")
    score = results[SELECTION_COLUMN].max()
    logger.info("%s : val mAP50-95 %.4f", run_name, score)
    return {"run_id": run_id, "config": run_name, "run_dir": runs_dir / run_name, "val_map50_95": float(score)}


def train_detector(quick: bool = False) -> dict:
    pipeline_config = load_config("pipeline")["detection"]
    train_args = load_config("detection")
    train_args.pop("model")
    train_args["data"] = str(project_path(train_args["data"]))
    if quick:
        train_args.update(pipeline_config["quick"])

    setup_mlflow(pipeline_config["experiment"])
    settings.update({"mlflow": True})
    runs_dir = project_path(pipeline_config["runs_dir"])

    results = [train_candidate(weights, train_args, runs_dir, quick) for weights in pipeline_config["candidates"]]
    best = max(results, key=lambda r: r["val_map50_95"])
    shutil.copy2(best["run_dir"] / "weights" / "best.pt", staging_dir() / "detector.pt")
    summary = {
        "run_id": best["run_id"],
        "config": best["config"],
        "class_names": load_config("data")["class_names"],
        "img_size": train_args["imgsz"],
        "quick": quick,
        "val_map50_95": best["val_map50_95"],
        "run_dir": str(best["run_dir"]),
    }
    write_json(staging_dir() / "detector.json", summary)
    logger.info("Détecteur retenu : %s (val mAP50-95 %.4f)", best["config"], best["val_map50_95"])
    return summary
