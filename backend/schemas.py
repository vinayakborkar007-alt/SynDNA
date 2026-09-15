from pydantic import BaseModel
from typing import List, Dict, Optional


class PredictionRequest(BaseModel):
    sequence: str
    host: str = "unknown"


class SuspiciousRegion(BaseModel):
    start: int
    end: int
    score: float


class PredictionResponse(BaseModel):
    success: bool
    sequence_length: int = 0
    anomaly_score: float = 0.0
    classification: str = "Unknown"
    suspicious_regions: List[SuspiciousRegion] = []
    feature_importance: Dict[str, float] = {}
    error: Optional[str] = None