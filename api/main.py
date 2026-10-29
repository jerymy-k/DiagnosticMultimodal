import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from api.config import (
    CLASSIFIER_META_PATH,
    CLASSIFIER_PATH,
    DETECTION_CONF,
    DETECTION_IOU,
    DETECTOR_META_PATH,
    DETECTOR_PATH,
)
from api.schemas import ClassificationResponse, DetectionResponse, ErrorResponse, HealthResponse
from api.services.classifier import Classifier
from api.services.detector import Detector
from api.services.images import read_image

logger = logging.getLogger("uvicorn.error")


def load_model(name, loader, model_path, meta_path):
    try:
        model = loader(model_path, meta_path)
    except Exception:
        logger.exception("Échec du chargement du modèle %s (%s)", name, model_path)
        return None
    logger.info("Modèle %s chargé (run MLflow %s)", name, model.meta["mlflow_run_id"])
    return model


def get_model(request: Request, name: str):
    model = request.app.state.models.get(name)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Le modèle {name} n'est pas chargé.")
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {
        "classifier": load_model("classifier", Classifier, CLASSIFIER_PATH, CLASSIFIER_META_PATH),
        "detector": load_model("detector", Detector, DETECTOR_PATH, DETECTOR_META_PATH),
    }
    yield
    app.state.models.clear()


app = FastAPI(
    title="Diagnostic Multimodal API",
    description="Classification (InceptionV3) et détection (YOLOv8) de tumeurs cérébrales sur IRM.",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def internal_error_handler(request: Request, exc: Exception):
    logger.exception("Erreur interne sur %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Erreur interne du serveur."})


IMAGE_ERRORS = {
    400: {"model": ErrorResponse, "description": "Image corrompue ou illisible",
          "content": {"application/json": {"example": {"detail": "Image corrompue ou illisible."}}}},
    415: {"model": ErrorResponse, "description": "Type de fichier non supporté",
          "content": {"application/json": {"example": {"detail": "Type de fichier non supporté (text/plain). Formats acceptés : JPEG, PNG."}}}},
    500: {"model": ErrorResponse, "description": "Erreur interne",
          "content": {"application/json": {"example": {"detail": "Erreur interne du serveur."}}}},
}


def model_error(name):
    return {503: {"model": ErrorResponse, "description": "Modèle non chargé",
                  "content": {"application/json": {"example": {"detail": f"Le modèle {name} n'est pas chargé."}}}}}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Santé"],
    summary="État de l'API",
    description="Indique si l'API répond et si chaque modèle a bien été chargé au démarrage.",
)
def health(request: Request):
    models_loaded = {name: model is not None for name, model in request.app.state.models.items()}
    return HealthResponse(status="ok" if all(models_loaded.values()) else "degraded", models_loaded=models_loaded)


@app.post(
    "/predict/classification",
    response_model=ClassificationResponse,
    tags=["Classification"],
    summary="Classer une IRM cérébrale",
    description=(
        "Reçoit une image IRM (JPEG ou PNG), la redimensionne en 299×299 comme à l'entraînement "
        "et renvoie la classe prédite (Glioma, Meningioma, No Tumor, Pituitary), sa confiance "
        "et les probabilités de chaque classe."
    ),
    responses={**IMAGE_ERRORS, **model_error("classifier")},
)
def predict_classification(request: Request, file: UploadFile = File(..., description="Image IRM (JPEG ou PNG)")):
    classifier = get_model(request, "classifier")
    probabilities = classifier.predict(read_image(file))
    predicted_class = max(probabilities, key=probabilities.get)
    return ClassificationResponse(
        predicted_class=predicted_class,
        confidence=probabilities[predicted_class],
        probabilities=probabilities,
    )


@app.post(
    "/predict/detection",
    response_model=DetectionResponse,
    response_model_exclude_none=True,
    tags=["Détection"],
    summary="Détecter et localiser une tumeur",
    description=(
        "Reçoit une image IRM (JPEG ou PNG) et renvoie les boîtes détectées par YOLOv8 : classe, confiance "
        "et coordonnées `[x1, y1, x2, y2]` en pixels. Une image sans détection renvoie une liste vide. "
        "Avec `annotated=true`, l'image annotée est ajoutée en base64 (JPEG)."
    ),
    responses={**IMAGE_ERRORS, **model_error("detector")},
)
def predict_detection(
    request: Request,
    file: UploadFile = File(..., description="Image IRM (JPEG ou PNG)"),
    conf: float = Query(DETECTION_CONF, ge=0, le=1, description="Seuil de confiance minimal"),
    iou: float = Query(DETECTION_IOU, ge=0, le=1, description="Seuil IoU de la NMS"),
    annotated: bool = Query(False, description="Ajouter l'image annotée en base64"),
):
    detector = get_model(request, "detector")
    detections, result = detector.predict(read_image(file), conf, iou)
    return DetectionResponse(
        detections=detections,
        count=len(detections),
        annotated_image=detector.annotate(result) if annotated else None,
    )
