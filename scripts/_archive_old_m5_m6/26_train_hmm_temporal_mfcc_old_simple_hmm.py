from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


MANIFEST_PATH = Path("data/processed/manifest.csv")

OUT_MODEL = Path("outputs/models/hmm_temporal_mfcc.npz")
OUT_METRICS_JSON = Path("outputs/metrics/hmm_temporal_mfcc_metrics.json")
OUT_METRICS_CSV = Path("outputs/metrics/hmm_temporal_mfcc_metrics.csv")
OUT_PREDS = Path("outputs/predictions/hmm_temporal_mfcc_test_predictions.csv")

SAMPLE_RATE = 16000
TARGET_SECONDS = 3
TARGET_NUM_SAMPLES = SAMPLE_RATE * TARGET_SECONDS

N_MFCC = 13
N_FFT = 400          # 25 ms at 16 kHz
HOP_LENGTH = 160    # 10 ms at 16 kHz
N_STATES = 3

EPS = 1e-6
NEG_INF = -1e12

LABELS = {
    0: "fluent",
    1: "stutter",
}


def ensure_dirs() -> None:
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)


def read_manifest() -> List[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

    rows = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "path" not in row or "label_id" not in row or "split" not in row:
                raise ValueError("manifest.csv must contain: path, label_id, split")
            rows.append(row)

    return rows


def prepare_audio(path: str) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if len(y) > TARGET_NUM_SAMPLES:
        y = y[:TARGET_NUM_SAMPLES]
    elif len(y) < TARGET_NUM_SAMPLES:
        y = np.pad(y, (0, TARGET_NUM_SAMPLES - len(y)), mode="constant")

    peak = np.max(np.abs(y)) + EPS
    y = y / peak

    return y.astype(np.float32)


def extract_mfcc_sequence(path: str) -> np.ndarray:
    """
    Returns frame-level feature sequence:
    [MFCC, delta MFCC, delta-delta MFCC]
    Shape: [T, 39]
    """
    import librosa

    y = prepare_audio(path)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    delta = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)

    feat = np.concatenate([mfcc, delta, delta2], axis=0).T
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    return feat.astype(np.float32)


def init_stats(feature_dim: int) -> Dict[int, Dict[str, np.ndarray]]:
    stats = {}

    for label_id in [0, 1]:
        stats[label_id] = {
            "sum": np.zeros((N_STATES, feature_dim), dtype=np.float64),
            "sumsq": np.zeros((N_STATES, feature_dim), dtype=np.float64),
            "count": np.zeros((N_STATES,), dtype=np.float64),
            "num_sequences": np.array(0, dtype=np.int64),
        }

    return stats


def update_hmm_stats(
    stats: Dict[int, Dict[str, np.ndarray]],
    label_id: int,
    sequence: np.ndarray,
) -> None:
    """
    Simple supervised HMM estimation:
    Split each sequence into N_STATES temporal regions and estimate
    diagonal Gaussian emission per state.
    """
    t = sequence.shape[0]
    stats[label_id]["num_sequences"] += 1

    for state in range(N_STATES):
        start = int(math.floor(state * t / N_STATES))
        end = int(math.floor((state + 1) * t / N_STATES))

        if end <= start:
            continue

        frames = sequence[start:end]

        stats[label_id]["sum"][state] += frames.sum(axis=0)
        stats[label_id]["sumsq"][state] += (frames ** 2).sum(axis=0)
        stats[label_id]["count"][state] += frames.shape[0]


def estimate_hmm_params(stats: Dict[int, Dict[str, np.ndarray]]) -> Dict[int, dict]:
    params = {}

    for label_id in [0, 1]:
        count = stats[label_id]["count"][:, None]
        count = np.maximum(count, 1.0)

        mean = stats[label_id]["sum"] / count
        var = (stats[label_id]["sumsq"] / count) - (mean ** 2)
        var = np.maximum(var, 1e-3)

        log_pi = np.full((N_STATES,), NEG_INF, dtype=np.float64)
        log_pi[0] = 0.0

        trans = np.zeros((N_STATES, N_STATES), dtype=np.float64)

        for s in range(N_STATES):
            if s < N_STATES - 1:
                trans[s, s] = 0.65
                trans[s, s + 1] = 0.35
            else:
                trans[s, s] = 1.0

        log_trans = np.full_like(trans, NEG_INF)
        nonzero = trans > 0
        log_trans[nonzero] = np.log(trans[nonzero])

        params[label_id] = {
            "mean": mean,
            "var": var,
            "log_pi": log_pi,
            "log_trans": log_trans,
            "num_sequences": int(stats[label_id]["num_sequences"]),
        }

    return params


def logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    if m <= NEG_INF / 2:
        return float(NEG_INF)
    return float(m + np.log(np.sum(np.exp(x - m))))


def gaussian_log_emission(sequence: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """
    Diagonal Gaussian log probability.
    Returns shape [T, N_STATES].
    """
    t = sequence.shape[0]
    emissions = np.zeros((t, N_STATES), dtype=np.float64)

    for s in range(N_STATES):
        diff = sequence - mean[s]
        log_det = np.sum(np.log(2.0 * np.pi * var[s]))
        maha = np.sum((diff ** 2) / var[s], axis=1)
        emissions[:, s] = -0.5 * (log_det + maha)

    return emissions


def hmm_log_likelihood(sequence: np.ndarray, hmm: dict) -> float:
    mean = hmm["mean"]
    var = hmm["var"]
    log_pi = hmm["log_pi"]
    log_trans = hmm["log_trans"]

    emissions = gaussian_log_emission(sequence, mean, var)

    t = sequence.shape[0]
    alpha = np.full((t, N_STATES), NEG_INF, dtype=np.float64)

    alpha[0] = log_pi + emissions[0]

    for i in range(1, t):
        for s in range(N_STATES):
            alpha[i, s] = emissions[i, s] + logsumexp(alpha[i - 1] + log_trans[:, s])

    return logsumexp(alpha[-1])


def softmax_two(a: float, b: float) -> Tuple[float, float]:
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    z = ea + eb
    return ea / z, eb / z


def predict_one(sequence: np.ndarray, params: Dict[int, dict]) -> Tuple[int, float, float, float]:
    ll_fluent = hmm_log_likelihood(sequence, params[0])
    ll_stutter = hmm_log_likelihood(sequence, params[1])

    prob_fluent, prob_stutter = softmax_two(ll_fluent, ll_stutter)
    pred = 1 if ll_stutter >= ll_fluent else 0

    return pred, prob_stutter, ll_fluent, ll_stutter


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    tp = int(((y_true_np == 1) & (y_pred_np == 1)).sum())
    tn = int(((y_true_np == 0) & (y_pred_np == 0)).sum())
    fp = int(((y_true_np == 0) & (y_pred_np == 1)).sum())
    fn = int(((y_true_np == 1) & (y_pred_np == 0)).sum())

    accuracy = (tp + tn) / max(1, len(y_true))

    precision_stutter = tp / max(1, tp + fp)
    recall_stutter = tp / max(1, tp + fn)
    specificity_fluent = tn / max(1, tn + fp)

    f1_stutter = 2 * precision_stutter * recall_stutter / max(
        EPS,
        precision_stutter + recall_stutter,
    )

    precision_fluent = tn / max(1, tn + fn)
    f1_fluent = 2 * precision_fluent * specificity_fluent / max(
        EPS,
        precision_fluent + specificity_fluent,
    )

    macro_f1 = (f1_stutter + f1_fluent) / 2
    balanced_accuracy = (recall_stutter + specificity_fluent) / 2

    return {
        "accuracy": float(accuracy),
        "precision_stutter": float(precision_stutter),
        "recall_stutter_sensitivity": float(recall_stutter),
        "specificity_fluent": float(specificity_fluent),
        "f1_stutter": float(f1_stutter),
        "f1_fluent": float(f1_fluent),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_split(rows: List[dict], split: str, params: Dict[int, dict], save_predictions: bool = False) -> dict:
    split_rows = [r for r in rows if r["split"] == split]

    y_true = []
    y_pred = []
    pred_rows = []

    print(f"[INFO] Evaluating split={split}, n={len(split_rows)}")

    for i, row in enumerate(split_rows, start=1):
        path = row["path"]
        label_id = int(row["label_id"])

        try:
            seq = extract_mfcc_sequence(path)
            pred, prob_stutter, ll_fluent, ll_stutter = predict_one(seq, params)
        except Exception as exc:
            print(f"[WARN] Failed sample {path}: {exc}")
            continue

        y_true.append(label_id)
        y_pred.append(pred)

        if save_predictions:
            pred_rows.append({
                "path": path,
                "y_true": label_id,
                "y_pred": pred,
                "prob_stutter": prob_stutter,
                "ll_fluent": ll_fluent,
                "ll_stutter": ll_stutter,
            })

        if i % 500 == 0:
            print(f"[INFO] {split}: processed {i}/{len(split_rows)}")

    metrics = compute_metrics(y_true, y_pred)

    if save_predictions:
        with OUT_PREDS.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "path",
                    "y_true",
                    "y_pred",
                    "prob_stutter",
                    "ll_fluent",
                    "ll_stutter",
                ],
            )
            writer.writeheader()
            writer.writerows(pred_rows)

    return metrics


def save_model(params: Dict[int, dict]) -> None:
    np.savez(
        OUT_MODEL,
        fluent_mean=params[0]["mean"],
        fluent_var=params[0]["var"],
        fluent_log_pi=params[0]["log_pi"],
        fluent_log_trans=params[0]["log_trans"],
        stutter_mean=params[1]["mean"],
        stutter_var=params[1]["var"],
        stutter_log_pi=params[1]["log_pi"],
        stutter_log_trans=params[1]["log_trans"],
        labels=np.array(["fluent", "stutter"]),
        n_states=np.array(N_STATES),
        n_mfcc=np.array(N_MFCC),
        sample_rate=np.array(SAMPLE_RATE),
        target_num_samples=np.array(TARGET_NUM_SAMPLES),
    )


def save_metrics_csv(metrics: dict) -> None:
    with OUT_METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model",
            "split",
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "precision_stutter",
            "recall_stutter_sensitivity",
            "specificity_fluent",
            "f1_stutter",
            "f1_fluent",
            "tp",
            "tn",
            "fp",
            "fn",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for split in ["val", "test"]:
            row = {
                "model": "hmm_temporal_mfcc",
                "split": split,
                **metrics[split],
            }
            writer.writerow(row)


def main() -> None:
    ensure_dirs()

    rows = read_manifest()

    train_rows = [r for r in rows if r["split"] == "train"]

    print("[INFO] M6 HMM Temporal MFCC Baseline")
    print(f"[INFO] Train samples: {len(train_rows)}")
    print(f"[INFO] Val samples: {sum(1 for r in rows if r['split'] == 'val')}")
    print(f"[INFO] Test samples: {sum(1 for r in rows if r['split'] == 'test')}")

    first_seq = extract_mfcc_sequence(train_rows[0]["path"])
    feature_dim = first_seq.shape[1]

    print(f"[INFO] Feature dim: {feature_dim}")
    print(f"[INFO] HMM states per class: {N_STATES}")

    stats = init_stats(feature_dim)

    for i, row in enumerate(train_rows, start=1):
        path = row["path"]
        label_id = int(row["label_id"])

        try:
            seq = extract_mfcc_sequence(path)
            update_hmm_stats(stats, label_id, seq)
        except Exception as exc:
            print(f"[WARN] Failed train sample {path}: {exc}")

        if i % 500 == 0:
            print(f"[INFO] Training stats: processed {i}/{len(train_rows)}")

    params = estimate_hmm_params(stats)
    save_model(params)

    val_metrics = evaluate_split(rows, "val", params, save_predictions=False)
    test_metrics = evaluate_split(rows, "test", params, save_predictions=True)

    metrics = {
        "model": "hmm_temporal_mfcc",
        "member_task": "M6 Classical Baseline 2 - HMM Temporal Model",
        "features": "MFCC + delta + delta-delta frame sequence",
        "hmm_type": "left-to-right HMM with diagonal Gaussian emissions",
        "n_states": N_STATES,
        "n_mfcc": N_MFCC,
        "sample_rate": SAMPLE_RATE,
        "target_seconds": TARGET_SECONDS,
        "val": val_metrics,
        "test": test_metrics,
    }

    with OUT_METRICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_metrics_csv(metrics)

    print("\n[OK] Saved model:", OUT_MODEL)
    print("[OK] Saved metrics JSON:", OUT_METRICS_JSON)
    print("[OK] Saved metrics CSV:", OUT_METRICS_CSV)
    print("[OK] Saved test predictions:", OUT_PREDS)

    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
