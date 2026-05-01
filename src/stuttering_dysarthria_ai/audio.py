from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is missing. Install only if needed: uv pip install librosa soundfile"
        ) from exc

    y, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return y.astype(np.float32)


def peak_normalize(y: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(y))) + 1e-8
    return (y / peak).astype(np.float32)


def pad_or_trim(y: np.ndarray, target_num_samples: int) -> np.ndarray:
    if len(y) > target_num_samples:
        return y[:target_num_samples].astype(np.float32)

    if len(y) < target_num_samples:
        y = np.pad(y, (0, target_num_samples - len(y)), mode="constant")

    return y.astype(np.float32)


def prepare_audio(
    path: str | Path,
    sample_rate: int = 16000,
    target_num_samples: int = 48000,
    normalize_peak: bool = True,
) -> np.ndarray:
    y = load_audio(path, sample_rate=sample_rate)
    y = pad_or_trim(y, target_num_samples)

    if normalize_peak:
        y = peak_normalize(y)

    return y.astype(np.float32)
