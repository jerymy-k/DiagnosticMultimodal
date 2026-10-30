import logging

import yaml

from src.data.prepare import is_readable
from src.data.yolo_labels import read_labels
from src.utils.config import load_config, project_path

logger = logging.getLogger(__name__)

CLASSIFICATION_SETS = ["train", "val", "test"]
YOLO_SETS = ["train", "valid", "test"]


def validate_classification(classification_dir, class_names):
    problems, counts = [], {}
    for set_name in CLASSIFICATION_SETS:
        set_dir = classification_dir / set_name
        found = sorted(p.name for p in set_dir.iterdir() if p.is_dir()) if set_dir.is_dir() else []
        if found != sorted(class_names):
            problems.append(f"classification/{set_name} : classes {found} au lieu de {class_names}")
            continue
        for class_name in class_names:
            images = list((set_dir / class_name).glob("*"))
            counts[f"{set_name}/{class_name}"] = len(images)
            if not images:
                problems.append(f"classification/{set_name}/{class_name} est vide")
            problems += [f"image illisible : {p}" for p in images if not is_readable(p)]
    return problems, counts


def validate_yolo(yolo_dir, class_names):
    problems, counts = [], {}
    data = yaml.safe_load((yolo_dir / "data.yaml").read_text())
    if data["nc"] != len(class_names) or data["names"] != class_names:
        problems.append(f"data.yaml incohérent : nc={data['nc']}, names={data['names']}")
    for set_name in YOLO_SETS:
        images = {p.stem for p in (yolo_dir / "images" / set_name).iterdir()}
        labels = {p.stem: p for p in (yolo_dir / "labels" / set_name).glob("*.txt")}
        counts[set_name] = len(images)
        problems += [f"yolo/{set_name} : image sans label {s}" for s in sorted(images - labels.keys())]
        problems += [f"yolo/{set_name} : label sans image {s}" for s in sorted(labels.keys() - images)]
        for stem, label_path in labels.items():
            boxes, invalid = read_labels(label_path, len(class_names))
            if invalid:
                problems.append(f"yolo/{set_name}/{stem} : {len(invalid)} lignes invalides")
            if not boxes:
                problems.append(f"yolo/{set_name}/{stem} : aucune boîte")
    return problems, counts


def validate_data() -> dict:
    config = load_config("data")
    processed_dir = project_path(config["processed_dir"])
    class_names = config["class_names"]

    cls_problems, cls_counts = validate_classification(processed_dir / "classification", class_names)
    yolo_problems, yolo_counts = validate_yolo(processed_dir / "yolo", class_names)
    logger.info("Classification : %s", cls_counts)
    logger.info("YOLO : %s", yolo_counts)

    problems = cls_problems + yolo_problems
    if problems:
        for problem in problems[:20]:
            logger.error(problem)
        raise ValueError(f"Validation des données échouée : {len(problems)} problèmes")
    logger.info("Données valides")
    return {"classification": cls_counts, "yolo": yolo_counts}
