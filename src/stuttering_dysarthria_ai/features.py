from __future__ import annotations

import numpy as np


def extract_logmel(
    y: np.ndarray,
    sr: int = 16000,
    n_mels: int = 64,
    n_fft: int = 400,
    hop_length: int = 160,
    fmin: int = 50,
    fmax: int = 8000,
) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    logmel = librosa.power_to_db(mel, ref=np.max)

    # Same normalization style used during feature caching/training.
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)

    return logmel.astype(np.float32)


def fix_logmel_shape(logmel: np.ndarray, expected_mels: int = 64, expected_frames: int = 301) -> np.ndarray:
    """
    Ensures inference input has exactly 64 x 301 shape.
    """
    if logmel.shape[0] != expected_mels:
        raise ValueError(f"Expected {expected_mels} mel bins, got {logmel.shape[0]}")

    current_frames = logmel.shape[1]

    if current_frames > expected_frames:
        logmel = logmel[:, :expected_frames]

    elif current_frames < expected_frames:
        pad_width = expected_frames - current_frames
        logmel = np.pad(logmel, ((0, 0), (0, pad_width)), mode="constant")

    return logmel.astype(np.float32)
