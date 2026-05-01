from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


MODEL_NAME = "facebook/wav2vec2-base"
CHECKPOINT_PATH = Path("outputs/models/wav2vec2_full_attention_test.pt")

SAMPLE_RATE = 16000
TARGET_SECONDS = 3
TARGET_LEN = SAMPLE_RATE * TARGET_SECONDS

THRESHOLD = 0.25

WARNING = (
    "Academic research prototype only. "
    "This is not a medical diagnosis tool."
)


def load_audio_for_wav2vec2(audio_path: str | Path) -> np.ndarray:
    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    elif len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)), mode="constant")

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak

    return y.astype(np.float32)


class Wav2Vec2FullAttentionPredictor:
    def __init__(self, checkpoint_path: str | Path = CHECKPOINT_PATH):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {self.checkpoint_path}")

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            label2id={"fluent": 0, "stutter": 1},
            id2label={0: "fluent", 1: "stutter"},
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_from_path(self, audio_path: str | Path) -> Dict[str, Any]:
        start = time.perf_counter()

        y = load_audio_for_wav2vec2(audio_path)

        inputs = self.feature_extractor(
            [y],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs["input_values"].to(self.device)
        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_values=input_values,
                attention_mask=attention_mask,
            )
            probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()

        prob_fluent = float(probs[0])
        prob_stutter = float(probs[1])

        prediction = "stutter" if prob_stutter >= THRESHOLD else "fluent"
        confidence = prob_stutter if prediction == "stutter" else prob_fluent

        elapsed = time.perf_counter() - start

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": {
                "fluent": prob_fluent,
                "stutter": prob_stutter,
            },
            "prob_stutter": prob_stutter,
            "threshold": THRESHOLD,
            "model_name": "Wav2Vec2 Full Attention",
            "base_model": MODEL_NAME,
            "model_version": "v1.0.0-final",
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": float(elapsed),
            "warning": WARNING,
        }

    def predict_from_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.predict_from_path(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
