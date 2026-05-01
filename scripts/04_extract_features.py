from pathlib import Path
import csv

import librosa
import numpy as np

MANIFEST_PATH = Path("data/processed/manifest.csv")
OUT_DIR = Path("data/processed/features")

SAMPLE_RATE = 16000
TARGET_SECONDS = 3
TARGET_LEN = SAMPLE_RATE * TARGET_SECONDS

N_MFCC = 40
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160


def load_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if len(y) > TARGET_LEN:
        y = y[:TARGET_LEN]
    elif len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))

    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak

    return y.astype(np.float32)


def extract_mfcc_stats(y):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mfcc, delta, delta2], axis=0)

    mean = features.mean(axis=1)
    std = features.std(axis=1)
    min_v = features.min(axis=1)
    max_v = features.max(axis=1)

    stats = np.concatenate([mean, std, min_v, max_v], axis=0)

    return stats.astype(np.float32)


def extract_logmel(y):
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

    # Normalize per clip
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)

    return logmel.astype(np.float32)


def read_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def process_split(rows, split_name):
    split_rows = [row for row in rows if row["split"] == split_name]

    mfcc_stats_list = []
    logmel_list = []
    labels = []
    paths = []

    total = len(split_rows)

    for i, row in enumerate(split_rows, start=1):
        audio_path = row["path"]
        label_id = int(row["label_id"])

        try:
            y = load_audio(audio_path)

            mfcc_stats = extract_mfcc_stats(y)
            logmel = extract_logmel(y)

            mfcc_stats_list.append(mfcc_stats)
            logmel_list.append(logmel)
            labels.append(label_id)
            paths.append(audio_path)

        except Exception as e:
            print(f"[SKIP] {audio_path} because {e}")

        if i % 500 == 0:
            print(f"[{split_name}] processed {i}/{total}")

    mfcc_stats_array = np.stack(mfcc_stats_list)
    logmel_array = np.stack(logmel_list)
    labels_array = np.array(labels, dtype=np.int64)
    paths_array = np.array(paths)

    out_path = OUT_DIR / f"{split_name}_features.npz"

    np.savez_compressed(
        out_path,
        mfcc_stats=mfcc_stats_array,
        logmel=logmel_array,
        labels=labels_array,
        paths=paths_array,
    )

    print(f"\n[OK] Saved {out_path}")
    print("mfcc_stats:", mfcc_stats_array.shape)
    print("logmel:", logmel_array.shape)
    print("labels:", labels_array.shape)


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("Missing data/processed/manifest.csv")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_manifest()

    for split_name in ["train", "val", "test"]:
        process_split(rows, split_name)


if __name__ == "__main__":
    main()