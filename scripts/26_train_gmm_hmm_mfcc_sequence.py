from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


MANIFEST_PATH = Path("data/processed/manifest.csv")

OUT_MODEL = Path("outputs/models/gmm_hmm_mfcc_sequence.npz")
OUT_METRICS_JSON = Path("outputs/metrics/gmm_hmm_mfcc_sequence_metrics.json")
OUT_METRICS_CSV = Path("outputs/metrics/gmm_hmm_mfcc_sequence_metrics.csv")
OUT_PREDS = Path("outputs/predictions/gmm_hmm_mfcc_sequence_test_predictions.csv")

SAMPLE_RATE = 16000
TARGET_SECONDS = 3
TARGET_NUM_SAMPLES = SAMPLE_RATE * TARGET_SECONDS

N_MFCC = 13
N_FFT = 400
HOP_LENGTH = 160

N_STATES = 3
N_MIXTURES = 4
MAX_FRAMES_PER_CLASS_STATE = 60000
MAX_EM_ITER = 25

SEED = 42
EPS = 1e-6
NEG_INF = -1e12


def ensure_dirs():
    for p in [OUT_MODEL, OUT_METRICS_JSON, OUT_METRICS_CSV, OUT_PREDS]:
        p.parent.mkdir(parents=True, exist_ok=True)


def read_manifest():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prepare_audio(path: str):
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if len(y) > TARGET_NUM_SAMPLES:
        y = y[:TARGET_NUM_SAMPLES]
    elif len(y) < TARGET_NUM_SAMPLES:
        y = np.pad(y, (0, TARGET_NUM_SAMPLES - len(y)), mode="constant")

    peak = np.max(np.abs(y)) + EPS
    y = y / peak
    return y.astype(np.float32)


def extract_sequence(path: str):
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

    x = np.concatenate([mfcc, delta, delta2], axis=0).T
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.astype(np.float32)


def split_state_frames(seq):
    t = seq.shape[0]
    parts = []

    for s in range(N_STATES):
        start = int(math.floor(s * t / N_STATES))
        end = int(math.floor((s + 1) * t / N_STATES))
        if end <= start:
            end = min(t, start + 1)
        parts.append(seq[start:end])

    return parts


def reservoir_add(store: list, frames: np.ndarray, max_items: int, rng: random.Random):
    for frame in frames:
        if len(store) < max_items:
            store.append(frame.astype(np.float32))
        else:
            j = rng.randint(0, max_items - 1)
            if rng.random() < 0.10:
                store[j] = frame.astype(np.float32)


def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-12)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def diag_gaussian_logpdf(X, means, vars_):
    diff = X[:, None, :] - means[None, :, :]
    log_det = np.sum(np.log(2.0 * np.pi * vars_), axis=1)
    maha = np.sum((diff ** 2) / vars_[None, :, :], axis=2)
    return -0.5 * (log_det[None, :] + maha)


def train_gmm(X, k, seed):
    rng = np.random.default_rng(seed)
    X = X.astype(np.float64)
    n, d = X.shape

    idx = rng.choice(n, size=k, replace=n < k)
    means = X[idx].copy()
    vars_ = np.tile(np.var(X, axis=0, keepdims=True) + 1e-3, (k, 1))
    weights = np.ones(k) / k

    prev = None

    for _ in range(MAX_EM_ITER):
        logp = diag_gaussian_logpdf(X, means, vars_) + np.log(weights[None, :] + 1e-12)
        logn = logsumexp(logp, axis=1, keepdims=True)
        resp = np.exp(logp - logn)

        Nk = resp.sum(axis=0) + 1e-8
        weights = Nk / n
        means = (resp.T @ X) / Nk[:, None]

        for j in range(k):
            diff = X - means[j]
            vars_[j] = (resp[:, j:j+1] * diff * diff).sum(axis=0) / Nk[j]

        vars_ = np.maximum(vars_, 1e-4)

        ll = float(logn.mean())
        if prev is not None and abs(ll - prev) < 1e-4:
            break
        prev = ll

    return weights.astype(np.float64), means.astype(np.float64), vars_.astype(np.float64)


def compute_global_norm(frame_bags):
    all_frames = []
    for c in [0, 1]:
        for s in range(N_STATES):
            all_frames.append(np.asarray(frame_bags[c][s], dtype=np.float32))

    X = np.concatenate(all_frames, axis=0)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    return mean, std


def build_transition():
    trans = np.zeros((N_STATES, N_STATES), dtype=np.float64)

    for s in range(N_STATES):
        if s < N_STATES - 1:
            trans[s, s] = 0.65
            trans[s, s + 1] = 0.35
        else:
            trans[s, s] = 1.0

    log_trans = np.full_like(trans, NEG_INF)
    mask = trans > 0
    log_trans[mask] = np.log(trans[mask])
    return log_trans


def train_gmm_hmm(rows):
    rng = random.Random(SEED)
    train_rows = [r for r in rows if r["split"] == "train"]

    frame_bags = {
        0: {s: [] for s in range(N_STATES)},
        1: {s: [] for s in range(N_STATES)},
    }

    print("[INFO] Collecting frame-level MFCC sequences for M6 GMM-HMM")

    for i, row in enumerate(train_rows, start=1):
        label = int(row["label_id"])

        try:
            seq = extract_sequence(row["path"])
            parts = split_state_frames(seq)

            for s, frames in enumerate(parts):
                reservoir_add(frame_bags[label][s], frames, MAX_FRAMES_PER_CLASS_STATE, rng)

        except Exception as exc:
            print("[WARN] failed:", row.get("path"), exc)

        if i % 500 == 0:
            print(f"[INFO] collected from {i}/{len(train_rows)} clips")

    mean, std = compute_global_norm(frame_bags)

    weights = np.zeros((2, N_STATES, N_MIXTURES), dtype=np.float64)
    means = None
    vars_ = None

    for c in [0, 1]:
        for s in range(N_STATES):
            X = np.asarray(frame_bags[c][s], dtype=np.float32)
            X = (X - mean) / std

            print(f"[INFO] Training GMM emission: class={c}, state={s}, frames={len(X)}")
            w, m, v = train_gmm(X, N_MIXTURES, seed=SEED + c * 10 + s)

            if means is None:
                d = m.shape[1]
                means = np.zeros((2, N_STATES, N_MIXTURES, d), dtype=np.float64)
                vars_ = np.zeros((2, N_STATES, N_MIXTURES, d), dtype=np.float64)

            weights[c, s] = w
            means[c, s] = m
            vars_[c, s] = v

    log_pi = np.full((2, N_STATES), NEG_INF, dtype=np.float64)
    log_pi[:, 0] = 0.0

    log_trans = np.stack([build_transition(), build_transition()], axis=0)

    return {
        "weights": weights,
        "means": means,
        "vars": vars_,
        "mean_norm": mean.astype(np.float64),
        "std_norm": std.astype(np.float64),
        "log_pi": log_pi,
        "log_trans": log_trans,
    }


def emission_logprob(seq, c, params):
    # seq standardized: [T, D]
    out = np.zeros((seq.shape[0], N_STATES), dtype=np.float64)

    for s in range(N_STATES):
        logp = diag_gaussian_logpdf(seq, params["means"][c, s], params["vars"][c, s])
        logp = logp + np.log(params["weights"][c, s][None, :] + 1e-12)
        out[:, s] = logsumexp(logp, axis=1)

    return out


def forward_ll(seq, c, params):
    emissions = emission_logprob(seq, c, params)
    log_pi = params["log_pi"][c]
    log_trans = params["log_trans"][c]

    t = emissions.shape[0]
    alpha = np.full((t, N_STATES), NEG_INF, dtype=np.float64)
    alpha[0] = log_pi + emissions[0]

    for i in range(1, t):
        for s in range(N_STATES):
            alpha[i, s] = emissions[i, s] + logsumexp(alpha[i - 1] + log_trans[:, s])

    return float(logsumexp(alpha[-1]))


def prob_stutter(ll0, ll1):
    m = max(ll0, ll1)
    a = math.exp(ll0 - m)
    b = math.exp(ll1 - m)
    return b / (a + b + 1e-12)


def predict_one(path, params):
    seq = extract_sequence(path)
    seq = (seq - params["mean_norm"]) / params["std_norm"]

    ll0 = forward_ll(seq, 0, params)
    ll1 = forward_ll(seq, 1, params)
    ps = prob_stutter(ll0, ll1)
    pred = 1 if ps >= 0.5 else 0

    return pred, ps, ll0, ll1


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, len(y_true))
    prec_s = tp / max(1, tp + fp)
    rec_s = tp / max(1, tp + fn)
    spec_f = tn / max(1, tn + fp)
    f1_s = 2 * prec_s * rec_s / max(EPS, prec_s + rec_s)

    prec_f = tn / max(1, tn + fn)
    f1_f = 2 * prec_f * spec_f / max(EPS, prec_f + spec_f)

    return {
        "accuracy": float(acc),
        "macro_f1": float((f1_s + f1_f) / 2),
        "balanced_accuracy": float((rec_s + spec_f) / 2),
        "precision_stutter": float(prec_s),
        "recall_stutter_sensitivity": float(rec_s),
        "specificity_fluent": float(spec_f),
        "f1_stutter": float(f1_s),
        "f1_fluent": float(f1_f),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def tune_threshold(y_true, probs):
    best = None

    for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = (np.asarray(probs) >= thr).astype(np.int64)
        m = compute_metrics(y_true, pred)
        m["threshold"] = float(thr)

        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = m

    return float(best["threshold"]), best


def eval_split(rows, split, params, threshold=None, save=False):
    split_rows = [r for r in rows if r["split"] == split]

    y_true = []
    probs = []
    raw = []

    print(f"[INFO] Evaluating {split}, n={len(split_rows)}")

    for i, row in enumerate(split_rows, start=1):
        try:
            pred0, ps, ll0, ll1 = predict_one(row["path"], params)
        except Exception as exc:
            print("[WARN] failed:", row.get("path"), exc)
            continue

        y_true.append(int(row["label_id"]))
        probs.append(float(ps))
        raw.append((row["path"], int(row["label_id"]), ps, ll0, ll1))

        if i % 500 == 0:
            print(f"[INFO] {split}: {i}/{len(split_rows)}")

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    if threshold is None:
        threshold, metrics = tune_threshold(y_true, probs)
    else:
        pred = (probs >= threshold).astype(np.int64)
        metrics = compute_metrics(y_true, pred)
        metrics["threshold"] = float(threshold)

    if save:
        pred = (probs >= threshold).astype(np.int64)
        with OUT_PREDS.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "y_true", "y_pred", "prob_stutter", "ll_fluent", "ll_stutter"])
            for (p, yt, ps, ll0, ll1), yp in zip(raw, pred):
                writer.writerow([p, yt, int(yp), float(ps), float(ll0), float(ll1)])

    return metrics, threshold


def save_npz(params, threshold):
    np.savez(
        OUT_MODEL,
        weights=params["weights"],
        means=params["means"],
        vars=params["vars"],
        mean_norm=params["mean_norm"],
        std_norm=params["std_norm"],
        log_pi=params["log_pi"],
        log_trans=params["log_trans"],
        n_states=np.array(N_STATES),
        n_mixtures=np.array(N_MIXTURES),
        threshold=np.array(threshold),
        sample_rate=np.array(SAMPLE_RATE),
        target_num_samples=np.array(TARGET_NUM_SAMPLES),
    )


def save_csv(metrics):
    with OUT_METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model", "split", "accuracy", "macro_f1", "balanced_accuracy",
            "precision_stutter", "recall_stutter_sensitivity", "specificity_fluent",
            "f1_stutter", "f1_fluent", "tp", "tn", "fp", "fn", "threshold"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split in ["val", "test"]:
            writer.writerow({"model": "gmm_hmm_mfcc_sequence", "split": split, **metrics[split]})


def main():
    ensure_dirs()
    rows = read_manifest()

    print("[INFO] M6 GMM-HMM over MFCC sequences")
    print("[INFO] N_STATES:", N_STATES, "N_MIXTURES:", N_MIXTURES)

    params = train_gmm_hmm(rows)

    val_metrics, threshold = eval_split(rows, "val", params, threshold=None, save=False)
    test_metrics, _ = eval_split(rows, "test", params, threshold=threshold, save=True)

    save_npz(params, threshold)

    final = {
        "model": "gmm_hmm_mfcc_sequence",
        "member_task": "M6 Classical Baseline 2 - GMM-HMM Temporal Model",
        "features": "MFCC + delta + delta-delta frame sequence",
        "hmm_type": "left-to-right HMM",
        "emission_type": f"{N_MIXTURES}-component diagonal GMM per state",
        "n_states": N_STATES,
        "n_mixtures": N_MIXTURES,
        "threshold": threshold,
        "val": val_metrics,
        "test": test_metrics,
    }

    OUT_METRICS_JSON.write_text(json.dumps(final, indent=2), encoding="utf-8")
    save_csv({"val": val_metrics, "test": test_metrics})

    print("\n[OK] Saved model:", OUT_MODEL)
    print("[OK] Saved metrics:", OUT_METRICS_JSON)
    print("[OK] Saved predictions:", OUT_PREDS)
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
