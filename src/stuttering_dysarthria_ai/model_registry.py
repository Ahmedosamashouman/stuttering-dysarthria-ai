from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from stuttering_dysarthria_ai.models.cnn_bilstm_v2 import CNNBiLSTMAttentionV2


class ModelRegistry:
    def __init__(self, model_dir: str | Path = "outputs/production_model"):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config: Dict[str, Any] = {}
        self.labels: Dict[int, str] = {}
        self.model_info: Dict[str, Any] = {}
        self.model = None

    def load(self) -> None:
        config_path = self.model_dir / "config.json"
        labels_path = self.model_dir / "labels.json"
        info_path = self.model_dir / "model_info.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing config file: {config_path}")

        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        if not info_path.exists():
            raise FileNotFoundError(f"Missing model info file: {info_path}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))

        raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
        self.labels = {int(k): v for k, v in raw_labels.items()}

        self.model_info = json.loads(info_path.read_text(encoding="utf-8"))

        model_cfg = self.config["model"]
        checkpoint_path = self.model_dir / model_cfg["checkpoint"]

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        self.model = CNNBiLSTMAttentionV2(
            num_classes=int(model_cfg["num_classes"]),
            hidden_size=int(model_cfg["hidden_size"]),
            dropout=float(model_cfg["dropout"]),
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)
        self.model.eval()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None
