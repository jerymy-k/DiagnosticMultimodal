import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Query, Request, UploadFile
from PIL import Image

from api.config import (
    CLASSIFIER_META_PATH,
    CLASSIFIER_PATH,
    DETECTION_CONF,
    DETECTION_IOU,
    DETECTOR_META_PATH,
    DETECTOR_PATH,
)
from api.schemas import ClassificationResponse, DetectionResponse, HealthResponse
from api.services.classifier import Classifier
from api.services.detector import Detector

logger = logging.getLogger("uvicorn.error")


def load_model(name, loader, model_path, meta_path):
    try:
        model = loader(model_path, meta_path)
    except Exception:
        logger.exception("Échec du chargement du modèle %s (%s)", name, model_path)
        return None
    logger.info("Modèle %s chargé (run MLflow %s)", name, model.meta["mlflow_run_id"])
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {
        "classifier": load_model("classifier", Classifier, CLASSIFIER_PATH, CLASSIFIER_META_PATH),
        "detector": load_model("detector", Detector, DETECTOR_PATH, DETECTOR_META_PATH),
    }
    yield
    app.state.models.clear()


app = FastAPI(title="Diagnostic Multimodal API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    models_loaded = {name: model is not None for name, model in request.app.state.models.items()}
    return HealthResponse(status="ok" if all(models_loaded.values()) else "degraded", models_loaded=models_loaded)


@app.post("/predict/classification", response_model=ClassificationResponse)
def predict_classification(request: Request, file: UploadFile = File(...)):
    image = Image.open(file.file)
    probabilities = request.app.state.models["classifier"].predict(image)
    predicted_class = max(probabilities, key=probabilities.get)
    return ClassificationResponse(
        predicted_class=predicted_class,
        confidence=probabilities[predicted_class],
        probabilities=probabilities,
    )


@app.post("/predict/detection", response_model=DetectionResponse, response_model_exclude_none=True)
def predict_detection(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(DETECTION_CONF, ge=0, le=1),
    iou: float = Query(DETECTION_IOU, ge=0, le=1),
    annotated: bool = Query(False),
):
    detector = request.app.state.models["detector"]
    detections, result = detector.predict(Image.open(file.file), conf, iou)
    return DetectionResponse(
        detections=detections,
        count=len(detections),
        annotated_image=detector.annotate(result) if annotated else None,
    )
