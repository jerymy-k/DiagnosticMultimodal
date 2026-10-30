import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def get_predictions(model, dataset):
    y_true, y_pred = [], []
    for images, labels in dataset:
        y_true.append(labels.numpy())
        y_pred.append(model(images, training=False).numpy().argmax(axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def compute_metrics(y_true, y_pred, labels, prefix):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision_macro": precision,
        f"{prefix}_recall_macro": recall,
        f"{prefix}_f1_macro": f1,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(title)
    return fig


def plot_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for metric, ax in zip(["accuracy", "loss"], axes):
        ax.plot(history.history[metric], label="train")
        ax.plot(history.history[f"val_{metric}"], label="val")
        ax.set_title(f"{title} - {metric}")
        ax.set_xlabel("epoch")
        ax.legend()
    return fig
