from __future__ import annotations

from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stuttering_dysarthria_ai.ssl_full_attention_inference import (
    Wav2Vec2FullAttentionPredictor,
)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_name: str
    model_version: str
    threshold: float
    sample_rate: int
    duration_seconds: float
    inference_seconds: float
    device: str
    warning: str


app = FastAPI(
    title="Stuttering Dysarthria AI API",
    version="1.0.0",
    description="Academic speech screening API using Wav2Vec2 Full Attention.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Wav2Vec2FullAttentionPredictor | None = None


@app.on_event("startup")
def startup_event():
    global predictor
    predictor = Wav2Vec2FullAttentionPredictor(
        model_dir="outputs/production_model_ssl"
    )


@app.get("/")
def root():
    return {
        "service": "stuttering_dysarthria_ai",
        "model": "Wav2Vec2 Full Attention",
        "health": "/health",
        "predict": "/v1/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    loaded = predictor is not None
    return {
        "status": "ok" if loaded else "model_not_loaded",
        "model_loaded": loaded,
        "model_name": predictor.model_info["model_name"] if loaded else None,
        "model_version": predictor.model_info["model_version"] if loaded else None,
        "device": predictor.device if loaded else None,
    }


def get_suffix(filename: str) -> str:
    suffix = "." + filename.split(".")[-1].lower() if "." in filename else ".wav"
    allowed = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}. Use WAV for best results.",
        )

    return suffix


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    filename = file.filename or "speech.wav"
    suffix = get_suffix(filename)

    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(audio_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Upload <= 15 MB.")

    try:
        return predictor.predict_from_bytes(audio_bytes, suffix=suffix)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
