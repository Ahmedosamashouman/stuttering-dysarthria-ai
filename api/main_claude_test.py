from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stuttering_dysarthria_ai.high_confidence_claude_test import high_confidence_decision
from stuttering_dysarthria_ai.inference import SpeechPathologyPredictor
from stuttering_dysarthria_ai.ssl_inference_claude_test import SSLWav2Vec2Predictor


app = FastAPI(
    title="Stuttering Dysarthria AI API — Claude Test",
    version="3.0.0-claude-test",
    description="Isolated Claude-test API. Does not replace api/main.py.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WARNING = "Academic research prototype only. Not a medical diagnosis tool."

cnn_predictor: SpeechPathologyPredictor | None = None
ssl_predictor: SSLWav2Vec2Predictor | None = None


class HealthResponse(BaseModel):
    status: str
    cnn_loaded: bool
    ssl_loaded: bool
    high_confidence_endpoint_ready: bool
    note: str


@app.on_event("startup")
def startup_event():
    global cnn_predictor, ssl_predictor

    cnn_predictor = SpeechPathologyPredictor(model_dir="outputs/production_model")

    try:
        ssl_predictor = SSLWav2Vec2Predictor(
            checkpoint_path="outputs/models/wav2vec2_ssl_bigger_claude_test.pt"
        )
        print("[OK] Claude-test SSL predictor loaded")
    except Exception as exc:
        ssl_predictor = None
        print("[WARN] Claude-test SSL predictor not loaded:", exc)


@app.get("/")
def root():
    return {
        "service": "stuttering_dysarthria_ai_claude_test",
        "version": "3.0.0-claude-test",
        "health": "/health",
        "high_confidence_predict": "/v1/predict_high_confidence_claude_test",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    cnn_loaded = cnn_predictor is not None and cnn_predictor.registry.is_loaded
    ssl_loaded = ssl_predictor is not None

    return {
        "status": "ok" if cnn_loaded and ssl_loaded else "not_ready",
        "cnn_loaded": cnn_loaded,
        "ssl_loaded": ssl_loaded,
        "high_confidence_endpoint_ready": cnn_loaded and ssl_loaded,
        "note": "This is the isolated Claude-test API. It does not replace api/main.py.",
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


async def save_upload_to_temp(file: UploadFile) -> str:
    filename = file.filename or "audio.wav"
    suffix = validate_suffix(filename)

    audio_bytes = await file.read()

    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(audio_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Upload <= 15 MB.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        return tmp.name


@app.post("/v1/predict_high_confidence_claude_test")
async def predict_high_confidence_claude_test(file: UploadFile = File(...)):
    if cnn_predictor is None:
        raise HTTPException(status_code=503, detail="CNN model is not loaded.")

    if ssl_predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Claude-test SSL model is not loaded.",
        )

    tmp_path = await save_upload_to_temp(file)

    try:
        cnn_result = cnn_predictor.predict_from_path(tmp_path)
        ssl_result = ssl_predictor.predict_from_path(tmp_path)

        cnn_prob_stutter = float(cnn_result["probabilities"]["stutter"])
        ssl_prob_stutter = float(ssl_result["probabilities"]["stutter"])

        decision = high_confidence_decision(
            cnn_prob_stutter=cnn_prob_stutter,
            ssl_prob_stutter=ssl_prob_stutter,
        )

        return {
            **decision,
            "cnn_probability_stutter": cnn_prob_stutter,
            "ssl_probability_stutter": ssl_prob_stutter,
            "cnn_probabilities": cnn_result["probabilities"],
            "ssl_probabilities": ssl_result["probabilities"],
            "models": {
                "cnn": cnn_result.get("model_name", "CNN-BiLSTM-Attention"),
                "ssl": ssl_result.get("model_name", "facebook/wav2vec2-base"),
            },
            "warning": WARNING,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Claude-test inference failed: {exc}",
        )

    finally:
        Path(tmp_path).unlink(missing_ok=True)
