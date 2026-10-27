from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "class": "Glioma",
                "confidence": 0.98,
                "probabilities": {"Glioma": 0.98, "Meningioma": 0.01, "No Tumor": 0.0, "Pituitary": 0.01},
            }]
        },
    )

    predicted_class: str = Field(alias="class")
    confidence: float
    probabilities: dict[str, float]


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    detected_class: str = Field(alias="class")
    confidence: float
    bbox: list[float] = Field(min_length=4, max_length=4)


class DetectionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{
                "detections": [{"class": "Glioma", "confidence": 0.91, "bbox": [120.5, 98.0, 210.3, 176.8]}],
                "count": 1,
            }]
        },
    )

    detections: list[Detection]
    count: int
    annotated_image: str | None = None
