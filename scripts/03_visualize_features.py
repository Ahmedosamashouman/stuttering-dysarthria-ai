from pathlib import Path
import csv
import random

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

MANIFEST_PATH = Path("data/processed/manifest.csv")
OUT_DIR = Path("outputs/figures/visual_pipeline")

SAMPLE_RATE = 16000
N_MFCC = 40
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
SEED = 42


def load_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    target_len = SAMPLE_RATE * 3

    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak

    return y


def plot_pipeline(audio_path, label, out_path):
    y = load_audio(audio_path)

    stft = np.abs(
        librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
    )
    spectrogram_db = librosa.amplitude_to_db(stft, ref=np.max)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=50,
        fmax=8000,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    fig, axes = plt.subplots(4, 1, figsize=(13, 12))

    librosa.display.waveshow(y, sr=SAMPLE_RATE, ax=axes[0])
    axes[0].set_title(f"1) Waveform - {label}")

    librosa.display.specshow(
        spectrogram_db,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="hz",
        ax=axes[1],
    )
    axes[1].set_title("2) Spectrogram")

    librosa.display.specshow(
        logmel,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axes[2],
    )
    axes[2].set_title("3) Log-Mel Spectrogram")

    librosa.display.specshow(
        mfcc,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        x_axis="time",
        ax=axes[3],
    )
    axes[3].set_title("4) MFCC Heatmap")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Missing data/processed/manifest.csv")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    random.seed(SEED)

    fluent = [r for r in rows if r["label"] == "fluent"]
    stutter = [r for r in rows if r["label"] == "stutter"]

    selected = []
    selected += random.sample(fluent, min(2, len(fluent)))
    selected += random.sample(stutter, min(2, len(stutter)))

    for i, row in enumerate(selected):
        audio_path = row["path"]
        label = row["label"]
        out_path = OUT_DIR / f"{i:02d}_{label}.png"

        print(f"[PLOT] {audio_path}")
        plot_pipeline(audio_path, label, out_path)
        print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()