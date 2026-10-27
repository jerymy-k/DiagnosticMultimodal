from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
