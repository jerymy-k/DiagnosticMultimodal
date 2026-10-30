import logging

import mlflow
from sklearn.metrics import classification_report

from src.classification.metrics import compute_metrics, get_predictions, plot_confusion_matrix
from src.classification.model import load_split, tf
from src.utils.config import load_config, project_path, read_json, staging_dir, write_json
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


def evaluate_classifier() -> dict:
    config = load_config("classification")
    classification_dir = project_path(load_config("data")["processed_dir"]) / "classification"
    summary = read_json(staging_dir() / "classifier.json")
    setup_mlflow(config["experiment"])

    model = tf.keras.models.load_model(staging_dir() / "classifier.keras")
    test_ds = load_split(classification_dir, "test", config["img_size"], config["batch_size"], config["seed"], False)
    class_names = summary["class_names"]
    labels = list(range(len(class_names)))

    y_true, y_pred = get_predictions(model, test_ds.cache().prefetch(tf.data.AUTOTUNE))
    metrics = compute_metrics(y_true, y_pred, labels, "test")
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=3, zero_division=0)
    logger.info("Classification - test :\n%s", report)

    with mlflow.start_run(run_id=summary["run_id"]):
        mlflow.log_metrics(metrics)
        mlflow.log_figure(plot_confusion_matrix(y_true, y_pred, class_names, f"{summary['config']} - Test"),
                          f"confusion_matrix_{summary['config']}.png")
        mlflow.log_text(report, f"classification_report_{summary['config']}.txt")

    summary["test_metrics"] = metrics
    write_json(staging_dir() / "classifier.json", summary)
    return metrics
