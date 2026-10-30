import logging

import mlflow
import mlflow.keras

from src.classification.metrics import compute_metrics, get_predictions, plot_history
from src.classification.model import build_model, load_split, tf, unfreeze_from
from src.utils.config import load_config, project_path, staging_dir, write_json
from src.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


def make_params(config, phase, settings, quick):
    return {
        "phase": phase,
        "backbone": config["backbone"],
        "img_size": config["img_size"],
        "batch_size": config["batch_size"],
        "optimizer": "Adam",
        "learning_rate": settings["learning_rate"],
        "loss": config["loss"],
        "dropout": config["dropout"],
        "max_epochs": config["quick"]["max_epochs"] if quick else settings["max_epochs"],
        "early_stopping_patience": settings["patience"],
    }


def fit_and_log(model, params, run_name, train_ds, val_ds, labels, quick, extra_params=None):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
        loss=params["loss"],
        metrics=["accuracy"],
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=params["early_stopping_patience"], restore_best_weights=True
    )
    with mlflow.start_run(run_name=run_name, tags={"source": "pipeline", "quick": str(quick)}) as run:
        mlflow.log_params({**params, **(extra_params or {})})
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=params["max_epochs"],
            callbacks=[early_stopping, mlflow.keras.MlflowCallback()],
        )
        val_true, val_pred = get_predictions(model, val_ds)
        metrics = compute_metrics(val_true, val_pred, labels, "val")
        mlflow.log_metrics(metrics)
        mlflow.log_figure(plot_history(history, run_name), f"curves_{run_name}.png")
    logger.info("%s : val F1 macro %.4f", run_name, metrics["val_f1_macro"])
    return {"run_id": run.info.run_id, "run_name": run_name, "metrics": metrics, "weights": model.get_weights()}


def train_classifier(quick: bool = False) -> dict:
    config = load_config("classification")
    data_config = load_config("data")
    classification_dir = project_path(data_config["processed_dir"]) / "classification"
    tf.keras.utils.set_random_seed(config["seed"])
    setup_mlflow(config["experiment"])

    splits = {
        split: load_split(classification_dir, split, config["img_size"], config["batch_size"], config["seed"], split == "train")
        for split in ["train", "val"]
    }
    class_names = splits["train"].class_names
    labels = list(range(len(class_names)))
    train_ds = splits["train"].prefetch(tf.data.AUTOTUNE)
    val_ds = splits["val"].cache().prefetch(tf.data.AUTOTUNE)

    model, base_model = build_model(len(class_names), config["img_size"], config["dropout"])
    fe_params = make_params(config, "feature_extraction", config["feature_extraction"], quick)
    results = [fit_and_log(model, fe_params, "inceptionv3-feature-extraction", train_ds, val_ds, labels, quick)]
    fe_weights = results[0]["weights"]

    ft_params = make_params(config, "fine_tuning", config["fine_tuning"], quick)
    for block in config["fine_tuning"]["blocks"]:
        model.set_weights(fe_weights)
        n_trainable = unfreeze_from(model, base_model, block)
        results.append(fit_and_log(
            model, ft_params, f"inceptionv3-fine-tuning-{block}", train_ds, val_ds, labels, quick,
            {"unfreeze_from": block, "trainable_params": n_trainable},
        ))

    best = max(results, key=lambda r: r["metrics"]["val_f1_macro"])
    model.set_weights(best["weights"])
    model.save(staging_dir() / "classifier.keras")
    summary = {
        "run_id": best["run_id"],
        "config": best["run_name"],
        "class_names": class_names,
        "img_size": [config["img_size"], config["img_size"]],
        "quick": quick,
        "val_metrics": best["metrics"],
    }
    write_json(staging_dir() / "classifier.json", summary)
    logger.info("Classifieur retenu : %s (val F1 %.4f)", best["run_name"], best["metrics"]["val_f1_macro"])
    return summary
