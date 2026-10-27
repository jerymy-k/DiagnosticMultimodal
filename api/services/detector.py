import json
from pathlib import Path

from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: Path, meta_path: Path):
        self.meta = json.loads(meta_path.read_text())
        self.class_names = self.meta["class_names"]
        self.img_size = self.meta["img_size"]
        self.model = YOLO(model_path)
