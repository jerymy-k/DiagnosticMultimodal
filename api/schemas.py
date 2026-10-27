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
