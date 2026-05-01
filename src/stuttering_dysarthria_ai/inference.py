from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch

from stuttering_dysarthria_ai.audio import prepare_audio
from stuttering_dysarthria_ai.features import extract_logmel, fix_logmel_shape
from stuttering_dysarthria_ai.model_registry import ModelRegistry
from stuttering_dysarthria_ai.postprocess import probabilities_to_prediction


class SpeechPathologyPredictor:
    def __init__(self, model_dir: str | Path = "outputs/production_model"):
        self.registry = ModelRegistry(model_dir)
        self.registry.load()

    def predict_from_path(self, audio_path: str | Path) -> Dict[str, Any]:
        start_time = time.perf_counter()

        cfg = self.registry.config
        audio_cfg = cfg["audio"]
        feature_cfg = cfg["features"]
        model_cfg = cfg["model"]
        decision_cfg = cfg["decision"]

        y = prepare_audio(
            audio_path,
            sample_rate=int(audio_cfg["sample_rate"]),
            target_num_samples=int(audio_cfg["target_num_samples"]),
            normalize_peak=bool(audio_cfg["normalize_peak"]),
        )

        logmel = extract_logmel(
            y,
            sr=int(audio_cfg["sample_rate"]),
            n_mels=int(feature_cfg["n_mels"]),
            n_fft=int(feature_cfg["n_fft"]),
            hop_length=int(feature_cfg["hop_length"]),
            fmin=int(feature_cfg["fmin"]),
            fmax=int(feature_cfg["fmax"]),
        )

        expected_mels = int(model_cfg["input_shape"][1])
        expected_frames = int(model_cfg["input_shape"][2])
        logmel = fix_logmel_shape(logmel, expected_mels, expected_frames)

        x = torch.tensor(logmel, dtype=torch.float32)
        x = x.unsqueeze(0).unsqueeze(0).to(self.registry.device)

        with torch.no_grad():
            logits = self.registry.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        probabilities = {
            self.registry.labels[i]: float(probs[i])
            for i in range(len(probs))
        }

        threshold = float(decision_cfg["threshold"])
        positive_label = str(decision_cfg["positive_label"])

        prediction, confidence = probabilities_to_prediction(
            probabilities=probabilities,
            threshold=threshold,
            positive_label=positive_label,
        )

        elapsed = time.perf_counter() - start_time

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": probabilities,
            "model_name": self.registry.model_info.get("model_name", "unknown"),
            "model_version": self.registry.model_info.get("model_version", "unknown"),
            "duration_seconds": float(elapsed),
            "sample_rate": int(audio_cfg["sample_rate"]),
            "threshold": threshold,
            "warning": self.registry.model_info.get(
                "medical_warning",
                "Research prototype only. Not a medical diagnosis tool.",
            ),
        }

    def predict_from_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.predict_from_path(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
