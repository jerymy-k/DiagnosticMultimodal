import logging
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from src.data.yolo_labels import read_labels
from src.utils.config import PROJECT_ROOT, load_config, project_path

logger = logging.getLogger(__name__)

YOLO_SET_NAMES = {"train": "train", "val": "valid", "test": "test"}


def is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def list_raw_images(raw_dir: Path, extensions: set[str]) -> pd.DataFrame:
    rows = []
    for split_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for image_path in sorted((class_dir / "images").iterdir()):
                if image_path.suffix.lower() not in extensions:
                    continue
                label_path = class_dir / "labels" / f"{image_path.stem}.txt"
                rows.append({
                    "path": image_path,
                    "label_path": label_path if label_path.exists() else None,
                    "class": class_dir.name,
                    "readable": is_readable(image_path),
                })
    return pd.DataFrame(rows)


def split_dataset(images: pd.DataFrame, config: dict) -> pd.DataFrame:
    train_df, temp_df = train_test_split(
        images, test_size=config["test_size"], stratify=images["class"], random_state=config["seed"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=config["val_test_ratio"], stratify=temp_df["class"], random_state=config["seed"]
    )
    return pd.concat([train_df.assign(set="train"), val_df.assign(set="val"), test_df.assign(set="test")])


def copy_classification(split_df: pd.DataFrame, out_dir: Path) -> None:
    for path, cls, set_name in zip(split_df["path"], split_df["class"], split_df["set"]):
        dest = out_dir / set_name / cls
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / path.name)


def augment_train(train_dir: Path, seed: int, augmentation_names: list[str]) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    def blur(img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 2)))

    def noise(img):
        arr = np.array(img, dtype=np.float32) + rng.normal(0, 10, (img.height, img.width, 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def flip(img):
        return ImageOps.mirror(img)

    augmentations = {"blur": blur, "noise": noise, "flip": flip}
    augmentations = {name: augmentations[name] for name in augmentation_names}

    class_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    originals = {d: sorted(d.glob("*.jpg")) for d in class_dirs}
    target = max(len(files) for files in originals.values())
    added = {}
    for class_dir, files in originals.items():
        missing = target - len(files)
        for i in range(missing):
            src = random.choice(files)
            name, augment = random.choice(list(augmentations.items()))
            with Image.open(src) as img:
                augment(img.convert("RGB")).save(class_dir / f"{src.stem}_aug_{name}_{i}.jpg")
        added[class_dir.name] = max(missing, 0)
    return added


def prepare_yolo(split_df: pd.DataFrame, yolo_dir: Path, class_names: list[str]) -> dict[str, int]:
    class_ids = {name: i for i, name in enumerate(class_names)}
    kept = discarded = 0
    for image_path, label_path, cls, set_name in zip(
        split_df["path"], split_df["label_path"], split_df["class"], split_df["set"]
    ):
        if label_path is None:
            discarded += 1
            continue
        boxes, _ = read_labels(label_path, len(class_names))
        if not any(box[0] == class_ids[cls] for box in boxes):
            discarded += 1
            continue
        for kind, src in [("images", image_path), ("labels", label_path)]:
            dest = yolo_dir / kind / YOLO_SET_NAMES[set_name]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / src.name)
        kept += 1

    data_yaml = {
        "path": str(yolo_dir),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    (yolo_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))
    return {"kept": kept, "discarded": discarded}


def is_prepared(processed_dir: Path) -> bool:
    return all(
        (processed_dir / relative).exists()
        for relative in ["split.csv", "classification/train", "yolo/data.yaml"]
    )


def check_and_prepare_datasets(force: bool = False) -> dict:
    config = load_config("data")
    raw_dir = project_path(config["raw_dir"])
    processed_dir = project_path(config["processed_dir"])

    if not raw_dir.is_dir() or not any(raw_dir.iterdir()):
        raise FileNotFoundError(f"Données brutes introuvables : {raw_dir}")

    if is_prepared(processed_dir) and not force:
        logger.info("Données déjà préparées dans %s : étape ignorée (force=True pour régénérer)", processed_dir)
        return {"status": "skipped"}

    images = list_raw_images(raw_dir, set(config["image_extensions"]))
    unreadable = images[~images["readable"]]
    if len(unreadable):
        logger.warning("%d images illisibles écartées : %s", len(unreadable), unreadable["path"].map(str).tolist())
    images = images[images["readable"]]
    logger.info("%d images valides trouvées", len(images))

    split_df = split_dataset(images, config)
    for name in ["classification", "yolo"]:
        shutil.rmtree(processed_dir / name, ignore_errors=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    split_df.assign(path=split_df["path"].map(lambda p: str(p.relative_to(PROJECT_ROOT))))[
        ["path", "class", "set"]
    ].to_csv(processed_dir / "split.csv", index=False)
    logger.info("Split : %s", split_df["set"].value_counts().to_dict())

    copy_classification(split_df, processed_dir / "classification")
    added = augment_train(processed_dir / "classification" / "train", config["seed"], config["augmentations"])
    logger.info("Augmentation du train : %s", added)

    yolo = prepare_yolo(split_df, processed_dir / "yolo", config["class_names"])
    logger.info("YOLO : %d paires gardées, %d écartées", yolo["kept"], yolo["discarded"])
    return {"status": "prepared", "images": len(split_df), "augmented": added, "yolo": yolo}
