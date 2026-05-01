from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from stuttering_dysarthria_ai.full_attention_inference import (
    Wav2Vec2FullAttentionPredictor,
)


app = FastAPI(
    title="Stuttering Dysarthria AI API",
    version="4.0.0",
    description="Final FastAPI deployment using Wav2Vec2 Full Attention for speech pathology screening.",
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
        checkpoint_path="outputs/models/wav2vec2_full_attention_test.pt"
    )
    print("[OK] Wav2Vec2 Full Attention predictor loaded")


@app.get("/")
def root():
    return {
        "service": "stuttering_dysarthria_ai",
        "version": "4.0.0",
        "model": "Wav2Vec2 Full Attention",
        "health": "/health",
        "predict": "/v1/predict",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    loaded = predictor is not None and predictor.is_loaded

    return {
        "status": "ok" if loaded else "model_not_loaded",
        "model_loaded": loaded,
        "model_name": "Wav2Vec2 Full Attention" if loaded else None,
        "checkpoint": "outputs/models/wav2vec2_full_attention_test.pt",
    }


def validate_suffix(filename: str) -> str:
    suffix = "." + filename.split(".")[-1].lower() if "." in filename else ".wav"
    allowed_suffixes = [".wav", ".mp3", ".m4a", ".flac", ".ogg"]

    if suffix not in allowed_suffixes:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {suffix}. Use WAV for best results.",
        )

    return suffix


@app.post("/v1/predict")
async def predict_audio(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    filename = file.filename or "audio.wav"
    suffix = validate_suffix(filename)

    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(audio_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Upload <= 15 MB.")

    try:
        return predictor.predict_from_bytes(audio_bytes, suffix=suffix)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
