from pydantic import BaseModel
from typing import List, Optional

class PredictionResponse(BaseModel):
    crop: str
    disease: str
    confidence: float
    is_healthy: bool
    raw_label: str
    status: str

    

class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    crop: Optional[str] = None
    disease: Optional[str] = None
    confidence: Optional[float] = None
    history: Optional[List[ChatMessage]] = []

class ChatResponse(BaseModel):
    response: str
    source: str   # "groq" | "gemini" | "none"