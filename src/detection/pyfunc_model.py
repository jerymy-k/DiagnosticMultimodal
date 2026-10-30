import mlflow
import pandas as pd
from ultralytics import YOLO

DETECTION_COLUMNS = {"image_path": "string", "class": "string", "confidence": "double",
                     "x1": "double", "y1": "double", "x2": "double", "y2": "double"}


class YoloDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        rows = []
        for image_path in model_input["image_path"]:
            result = self.model.predict(image_path, verbose=False)[0]
            for cls, conf, box in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
                rows.append({"image_path": image_path, "class": result.names[int(cls)], "confidence": conf,
                             "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
        return pd.DataFrame(rows, columns=list(DETECTION_COLUMNS))


mlflow.models.set_model(YoloDetector())
