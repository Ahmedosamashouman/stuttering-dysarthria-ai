from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


MODEL_NAME = "facebook/wav2vec2-base"
CHECKPOINT_PATH = Path("outputs/models/wav2vec2_full_attention_test.pt")
METRICS_PATH = Path("outputs/metrics/wav2vec2_full_attention_test_metrics.json")

SAMPLE_RATE = 16000
TARGET_SECONDS = 3
TARGET_NUM_SAMPLES = SAMPLE_RATE * TARGET_SECONDS

LABELS = {
    0: "fluent",
    1: "stutter",
}

DEFAULT_THRESHOLD = 0.25


def load_audio_for_wav2vec2(audio_path: str | Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

    if len(y) > TARGET_NUM_SAMPLES:
        y = y[:TARGET_NUM_SAMPLES]
    elif len(y) < TARGET_NUM_SAMPLES:
        y = np.pad(y, (0, TARGET_NUM_SAMPLES - len(y)), mode="constant")

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak

    return y.astype(np.float32)


def get_threshold() -> float:
    if not METRICS_PATH.exists():
        return DEFAULT_THRESHOLD

    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

        for key in ["best_threshold_val", "threshold"]:
            if key in metrics:
                return float(metrics[key])

        if "test_val_threshold" in metrics and "threshold" in metrics["test_val_threshold"]:
            return float(metrics["test_val_threshold"]["threshold"])

        return DEFAULT_THRESHOLD

    except Exception:
        return DEFAULT_THRESHOLD


class Wav2Vec2DeploymentPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path = CHECKPOINT_PATH,
        model_name: str = MODEL_NAME,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {self.checkpoint_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = get_threshold()

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            label2id={"fluent": 0, "stutter": 1},
            id2label={0: "fluent", 1: "stutter"},
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def predict_from_path(self, audio_path: str | Path) -> Dict[str, Any]:
        started = time.perf_counter()

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

        if prob_stutter >= self.threshold:
            prediction = "stutter"
            confidence = prob_stutter
        else:
            prediction = "fluent"
            confidence = prob_fluent

        elapsed = time.perf_counter() - started

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": {
                "fluent": prob_fluent,
                "stutter": prob_stutter,
            },
            "threshold": float(self.threshold),
            "model_name": "Wav2Vec2 Full Attention",
            "base_model": self.model_name,
            "checkpoint": str(self.checkpoint_path),
            "sample_rate": SAMPLE_RATE,
            "target_seconds": TARGET_SECONDS,
            "inference_seconds": float(elapsed),
            "device": self.device,
            "warning": "Academic screening prototype only. Not a medical diagnosis tool.",
        }

    def predict_from_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = Path(tmp.name)

        try:
            return self.predict_from_path(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
