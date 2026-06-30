from pydantic import BaseModel

class PredictionResponse(BaseModel):
    crop: str
    disease: str
    confidence: float
    is_healthy: bool
    raw_label: str
    status: str