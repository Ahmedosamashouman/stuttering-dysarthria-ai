from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class Wav2Vec2FullAttentionPredictor:
    def __init__(self, model_dir: str | Path = "outputs/production_model_ssl"):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.info_path = self.model_dir / "model_info.json"
        if not self.info_path.exists():
            raise FileNotFoundError(f"Missing model info: {self.info_path}")

        self.model_info = json.loads(self.info_path.read_text(encoding="utf-8"))

        self.sample_rate = int(self.model_info["input_audio"]["sample_rate"])
        self.duration_seconds = float(self.model_info["input_audio"]["duration_seconds"])
        self.target_num_samples = int(self.sample_rate * self.duration_seconds)

        self.threshold = float(self.model_info["decision"]["threshold"])
        self.base_model = self.model_info["base_model"]

        checkpoint_name = self.model_info["checkpoint"]
        self.checkpoint_path = self.model_dir / checkpoint_name

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {self.checkpoint_path}")

        self.feature_extractor = None
        self.model = None

        self._load()

    def _load(self) -> None:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.base_model)

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.base_model,
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

        # Remove DataParallel prefix if it exists.
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
            cleaned[new_key] = v

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)

        if missing:
            print("[WARN] Missing keys while loading model:", missing[:10])
        if unexpected:
            print("[WARN] Unexpected keys while loading model:", unexpected[:10])

        self.model.to(self.device)
        self.model.eval()

        print(f"[OK] Loaded Wav2Vec2 Full Attention on device={self.device}")

    def _load_audio(self, audio_path: str | Path) -> np.ndarray:
        import librosa

        y, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

        if len(y) > self.target_num_samples:
            y = y[: self.target_num_samples]
        elif len(y) < self.target_num_samples:
            y = np.pad(y, (0, self.target_num_samples - len(y)), mode="constant")

        peak = np.max(np.abs(y)) + 1e-8
        y = y / peak

        return y.astype(np.float32)

    def predict_from_path(self, audio_path: str | Path) -> Dict[str, Any]:
        start = time.perf_counter()

        y = self._load_audio(audio_path)

        inputs = self.feature_extractor(
            [y],
            sampling_rate=self.sample_rate,
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
            probs = torch.softmax(outputs.logits, dim=1).squeeze(0).detach().cpu().numpy()

        prob_fluent = float(probs[0])
        prob_stutter = float(probs[1])

        if prob_stutter >= self.threshold:
            prediction = "stutter"
            confidence = prob_stutter
        else:
            prediction = "fluent"
            confidence = prob_fluent

        elapsed = time.perf_counter() - start

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probabilities": {
                "fluent": prob_fluent,
                "stutter": prob_stutter,
            },
            "model_name": self.model_info["model_name"],
            "model_version": self.model_info["model_version"],
            "threshold": self.threshold,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "inference_seconds": float(elapsed),
            "device": self.device,
            "warning": self.model_info.get(
                "medical_warning",
                "Academic prototype only. Not a medical diagnosis tool.",
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
