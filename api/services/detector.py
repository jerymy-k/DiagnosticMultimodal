import base64
import io
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path: Path, meta_path: Path):
        self.meta = json.loads(meta_path.read_text())
        self.class_names = self.meta["class_names"]
        self.img_size = self.meta["img_size"]
        self.model = YOLO(model_path)

    def predict(self, image: Image.Image, conf: float, iou: float):
        result = self.model.predict(image.convert("RGB"), imgsz=self.img_size, conf=conf, iou=iou, verbose=False)[0]
        detections = [
            {"class": result.names[int(cls)], "confidence": score, "bbox": [round(v, 1) for v in box]}
            for cls, score, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())
        ]
        return detections, result

    @staticmethod
    def annotate(result) -> str:
        buffer = io.BytesIO()
        Image.fromarray(result.plot()[..., ::-1]).save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode()
