import json
from pathlib import Path

from src.utils.gpu import setup_gpu


class Classifier:
    def __init__(self, model_path: Path, meta_path: Path):
        setup_gpu()
        import tensorflow as tf

        self.meta = json.loads(meta_path.read_text())
        self.class_names = self.meta["class_names"]
        self.img_size = tuple(self.meta["img_size"])
        self.model = tf.keras.models.load_model(model_path)
