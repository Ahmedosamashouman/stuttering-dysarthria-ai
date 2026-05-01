from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


MANIFEST = Path("data/processed/manifest.csv")

OUT_MODEL = Path("outputs/models/gmm_hmm_mfcc_sequence_torch_gpu_test.pt")
OUT_METRICS = Path("outputs/metrics/gmm_hmm_mfcc_sequence_torch_gpu_test_metrics.json")
OUT_PREDS = Path("outputs/predictions/gmm_hmm_mfcc_sequence_torch_gpu_test_predictions.csv")
OUT_CSV = Path("outputs/metrics/gmm_hmm_mfcc_sequence_torch_gpu_test_metrics.csv")

SAMPLE_RATE = 16000
TARGET_SECONDS = 3
TARGET_NUM_SAMPLES = SAMPLE_RATE * TARGET_SECONDS

N_MFCC = 13
N_FFT = 400
HOP_LENGTH = 160

N_STATES = 3
N_MIXTURES = 4
MAX_FRAMES_PER_CLASS_STATE = 60000
MAX_EM_ITER = 30

SEED = 42
EPS = 1e-6
NEG_INF = -1e12


def ensure_dirs():
    for p in [OUT_MODEL, OUT_METRICS, OUT_PREDS, OUT_CSV]:
        p.parent.mkdir(parents=True, exist_ok=True)


def read_manifest():
    with MANIFEST.open("r", encoding="utf-8") as f:
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
    out = []

    for s in range(N_STATES):
        start = int(math.floor(s * t / N_STATES))
        end = int(math.floor((s + 1) * t / N_STATES))
        if end <= start:
            end = min(t, start + 1)
        out.append(seq[start:end])

    return out


def reservoir_add(store, frames, max_items, rng):
    for frame in frames:
        if len(store) < max_items:
            store.append(frame.astype(np.float32))
        else:
            j = rng.randint(0, max_items - 1)
            if rng.random() < 0.10:
                store[j] = frame.astype(np.float32)


def diag_logpdf(x, means, vars_):
    diff = x[:, None, :] - means[None, :, :]
    log_det = torch.sum(torch.log(2.0 * torch.pi * vars_), dim=1)
    maha = torch.sum((diff * diff) / vars_[None, :, :], dim=2)
    return -0.5 * (log_det[None, :] + maha)


def train_gmm_torch(x, k, device, seed):
    torch.manual_seed(seed)

    x = torch.tensor(x, dtype=torch.float32, device=device)
    n, d = x.shape

    idx = torch.randperm(n, device=device)[:k]
    means = x[idx].clone()

    global_var = torch.var(x, dim=0, unbiased=False).clamp_min(1e-3)
    vars_ = global_var.unsqueeze(0).repeat(k, 1).clone()
    weights = torch.ones(k, dtype=torch.float32, device=device) / k

    prev = None

    for _ in range(MAX_EM_ITER):
        logp = diag_logpdf(x, means, vars_) + torch.log(weights[None, :] + 1e-12)
        logn = torch.logsumexp(logp, dim=1, keepdim=True)
        resp = torch.exp(logp - logn)

        nk = resp.sum(dim=0).clamp_min(1e-8)
        weights = nk / n
        means = (resp.T @ x) / nk[:, None]

        for j in range(k):
            diff = x - means[j]
            vars_[j] = (resp[:, j:j + 1] * diff * diff).sum(dim=0) / nk[j]

        vars_ = vars_.clamp_min(1e-4)

        ll = float(logn.mean().detach().cpu())
        if prev is not None and abs(ll - prev) < 1e-4:
            break
        prev = ll

    return {
        "weights": weights.detach().cpu(),
        "means": means.detach().cpu(),
        "vars": vars_.detach().cpu(),
    }


def collect_frame_bags(rows):
    rng = random.Random(SEED)
    train_rows = [r for r in rows if r["split"] == "train"]

    bags = {
        0: {s: [] for s in range(N_STATES)},
        1: {s: [] for s in range(N_STATES)},
    }

    print("[INFO] Collecting MFCC sequence frames")

    for i, row in enumerate(train_rows, start=1):
        label = int(row["label_id"])

        try:
            seq = extract_sequence(row["path"])
            parts = split_state_frames(seq)

            for s, frames in enumerate(parts):
                reservoir_add(bags[label][s], frames, MAX_FRAMES_PER_CLASS_STATE, rng)

        except Exception as exc:
            print("[WARN] failed:", row.get("path"), exc)

        if i % 500 == 0:
            print(f"[INFO] collected {i}/{len(train_rows)}")

    return bags


def compute_norm(bags):
    frames = []
    for c in [0, 1]:
        for s in range(N_STATES):
            frames.append(np.asarray(bags[c][s], dtype=np.float32))

    x = np.concatenate(frames, axis=0)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def build_transition():
    trans = torch.zeros(N_STATES, N_STATES, dtype=torch.float32)

    for s in range(N_STATES):
        if s < N_STATES - 1:
            trans[s, s] = 0.65
            trans[s, s + 1] = 0.35
        else:
            trans[s, s] = 1.0

    log_trans = torch.full_like(trans, NEG_INF)
    mask = trans > 0
    log_trans[mask] = torch.log(trans[mask])

    return log_trans


def train_model(rows, device):
    bags = collect_frame_bags(rows)
    mean, std = compute_norm(bags)

    model = {
        "weights": {},
        "means": {},
        "vars": {},
        "mean_norm": torch.tensor(mean),
        "std_norm": torch.tensor(std),
        "log_pi": torch.full((2, N_STATES), NEG_INF),
        "log_trans": torch.stack([build_transition(), build_transition()], dim=0),
    }

    model["log_pi"][:, 0] = 0.0

    for c in [0, 1]:
        for s in range(N_STATES):
            x = np.asarray(bags[c][s], dtype=np.float32)
            x = (x - mean) / std

            print(f"[INFO] Training GPU GMM emission class={c}, state={s}, frames={len(x)}")
            gmm = train_gmm_torch(x, N_MIXTURES, device, seed=SEED + c * 10 + s)

            model["weights"][(c, s)] = gmm["weights"]
            model["means"][(c, s)] = gmm["means"]
            model["vars"][(c, s)] = gmm["vars"]

    return model


def emission_logprob(seq_t, c, model, device):
    out = []

    for s in range(N_STATES):
        weights = model["weights"][(c, s)].to(device)
        means = model["means"][(c, s)].to(device)
        vars_ = model["vars"][(c, s)].to(device)

        logp = diag_logpdf(seq_t, means, vars_) + torch.log(weights[None, :] + 1e-12)
        out.append(torch.logsumexp(logp, dim=1))

    return torch.stack(out, dim=1)


def forward_ll(seq_np, c, model, device):
    mean = model["mean_norm"].to(device)
    std = model["std_norm"].to(device)

    seq = torch.tensor(seq_np, dtype=torch.float32, device=device)
    seq = (seq - mean) / std

    emissions = emission_logprob(seq, c, model, device)
    log_pi = model["log_pi"][c].to(device)
    log_trans = model["log_trans"][c].to(device)

    t = emissions.shape[0]
    alpha = torch.full((t, N_STATES), NEG_INF, dtype=torch.float32, device=device)
    alpha[0] = log_pi + emissions[0]

    for i in range(1, t):
        for s in range(N_STATES):
            alpha[i, s] = emissions[i, s] + torch.logsumexp(alpha[i - 1] + log_trans[:, s], dim=0)

    return float(torch.logsumexp(alpha[-1], dim=0).detach().cpu())


def prob_stutter(ll0, ll1):
    m = max(ll0, ll1)
    a = math.exp(ll0 - m)
    b = math.exp(ll1 - m)
    return b / (a + b + 1e-12)


def predict_one(path, model, device):
    seq = extract_sequence(path)
    ll0 = forward_ll(seq, 0, model, device)
    ll1 = forward_ll(seq, 1, model, device)
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


def eval_split(rows, split, model, device, threshold=None, save=False):
    split_rows = [r for r in rows if r["split"] == split]

    y_true = []
    probs = []
    raw = []

    print(f"[INFO] Evaluating {split}, n={len(split_rows)}")

    for i, row in enumerate(split_rows, start=1):
        try:
            _, ps, ll0, ll1 = predict_one(row["path"], model, device)
        except Exception as exc:
            print("[WARN] failed:", row.get("path"), exc)
            continue

        y_true.append(int(row["label_id"]))
        probs.append(float(ps))
        raw.append((row["path"], int(row["label_id"]), float(ps), float(ll0), float(ll1)))

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
            for (path, yt, ps, ll0, ll1), yp in zip(raw, pred):
                writer.writerow([path, yt, int(yp), ps, ll0, ll1])

    return metrics, threshold


def save_metrics_csv(metrics):
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "model", "split", "accuracy", "macro_f1", "balanced_accuracy",
            "precision_stutter", "recall_stutter_sensitivity", "specificity_fluent",
            "f1_stutter", "f1_fluent", "tp", "tn", "fp", "fn", "threshold"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for split in ["val", "test"]:
            writer.writerow({"model": "gmm_hmm_mfcc_sequence_torch_gpu_test", "split": split, **metrics[split]})


def main():
    ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    rows = read_manifest()

    model = train_model(rows, device)

    val_metrics, threshold = eval_split(rows, "val", model, device, threshold=None, save=False)
    test_metrics, _ = eval_split(rows, "test", model, device, threshold=threshold, save=True)

    torch.save(
        {
            "model": "gmm_hmm_mfcc_sequence_torch_gpu_test",
            "n_states": N_STATES,
            "n_mixtures": N_MIXTURES,
            "threshold": threshold,
            "state_dict": model,
        },
        OUT_MODEL,
    )

    final = {
        "model": "gmm_hmm_mfcc_sequence_torch_gpu_test",
        "member_task": "M6 GPU test - PyTorch GMM-HMM over MFCC sequences",
        "device": device,
        "features": "MFCC + delta + delta-delta frame sequence",
        "n_states": N_STATES,
        "n_mixtures": N_MIXTURES,
        "threshold": threshold,
        "val": val_metrics,
        "test": test_metrics,
    }

    OUT_METRICS.write_text(json.dumps(final, indent=2), encoding="utf-8")
    save_metrics_csv({"val": val_metrics, "test": test_metrics})

    print("\n[OK] Saved:", OUT_MODEL)
    print("[OK] Saved:", OUT_METRICS)
    print("[OK] Saved:", OUT_PREDS)
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
