from __future__ import annotations

from typing import Dict, Optional, Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_name: str
    model_version: str
    duration_seconds: float
    sample_rate: int
    threshold: float
    warning: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
