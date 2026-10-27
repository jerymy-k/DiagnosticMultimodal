import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.gpu import setup_gpu

setup_gpu()

import tensorflow as tf


class Classifier:
    def __init__(self, model_path: Path, meta_path: Path):
        self.meta = json.loads(meta_path.read_text())
        self.class_names = self.meta["class_names"]
        self.img_size = tuple(self.meta["img_size"])
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, image: Image.Image) -> tf.Tensor:
        pixels = tf.image.resize(np.asarray(image.convert("RGB"), dtype="float32"), self.img_size)
        return tf.expand_dims(pixels, 0)

    def predict(self, image: Image.Image) -> dict[str, float]:
        probabilities = self.model(self.preprocess(image), training=False).numpy()[0]
        return dict(zip(self.class_names, probabilities.tolist()))
