from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


TRAIN_FEATS = Path("data/processed/features/train_features.npz")
VAL_FEATS = Path("data/processed/features/val_features.npz")
TEST_FEATS = Path("data/processed/features/test_features.npz")

OUT_MODEL = Path("outputs/models/gmm_mfcc_vector_torch_gpu_test.pt")
OUT_METRICS = Path("outputs/metrics/gmm_mfcc_vector_torch_gpu_test_metrics.json")
OUT_PREDS = Path("outputs/predictions/gmm_mfcc_vector_torch_gpu_test_predictions.csv")
OUT_CSV = Path("outputs/metrics/gmm_mfcc_vector_torch_gpu_test_metrics.csv")

K_VALUES = [1, 2, 4, 8]
MAX_EM_ITER = 40
SEED = 42
EPS = 1e-6


def ensure_dirs():
    for p in [OUT_MODEL, OUT_METRICS, OUT_PREDS, OUT_CSV]:
        p.parent.mkdir(parents=True, exist_ok=True)


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    x = data["mfcc_stats"].astype(np.float32)
    y = data["labels"].astype(np.int64)

    if "paths" in data.files:
        paths = data["paths"]
    else:
        paths = np.array([f"sample_{i}" for i in range(len(y))], dtype=object)

    paths = np.array([str(p) for p in paths], dtype=object)
    return x, y, paths


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std, mean, std


def diag_logpdf_torch(x, means, vars_):
    # x: [N, D], means: [K, D], vars: [K, D]
    diff = x[:, None, :] - means[None, :, :]
    log_det = torch.sum(torch.log(2.0 * torch.pi * vars_), dim=1)
    maha = torch.sum((diff * diff) / vars_[None, :, :], dim=2)
    return -0.5 * (log_det[None, :] + maha)


def train_gmm_torch(x: torch.Tensor, k: int, seed: int):
    torch.manual_seed(seed)

    n, d = x.shape
    idx = torch.randperm(n, device=x.device)[:k]
    means = x[idx].clone()

    global_var = torch.var(x, dim=0, unbiased=False).clamp_min(1e-3)
    vars_ = global_var.unsqueeze(0).repeat(k, 1).clone()
    weights = torch.ones(k, device=x.device, dtype=torch.float32) / k

    prev_ll = None

    for _ in range(MAX_EM_ITER):
        log_probs = diag_logpdf_torch(x, means, vars_) + torch.log(weights[None, :] + 1e-12)
        log_norm = torch.logsumexp(log_probs, dim=1, keepdim=True)
        resp = torch.exp(log_probs - log_norm)

        nk = resp.sum(dim=0).clamp_min(1e-8)
        weights = nk / n
        means = (resp.T @ x) / nk[:, None]

        for j in range(k):
            diff = x - means[j]
            vars_[j] = (resp[:, j:j + 1] * diff * diff).sum(dim=0) / nk[j]

        vars_ = vars_.clamp_min(1e-4)

        ll = float(log_norm.mean().detach().cpu())
        if prev_ll is not None and abs(ll - prev_ll) < 1e-4:
            break
        prev_ll = ll

    return {
        "weights": weights.detach().cpu(),
        "means": means.detach().cpu(),
        "vars": vars_.detach().cpu(),
    }


def class_log_likelihood_torch(x: torch.Tensor, gmm: dict, device: str, batch_size: int = 4096):
    weights = gmm["weights"].to(device)
    means = gmm["means"].to(device)
    vars_ = gmm["vars"].to(device)

    outs = []

    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start + batch_size]
        log_probs = diag_logpdf_torch(xb, means, vars_) + torch.log(weights[None, :] + 1e-12)
        ll = torch.logsumexp(log_probs, dim=1)
        outs.append(ll.detach().cpu())

    return torch.cat(outs).numpy()


def prob_stutter(ll0, ll1):
    m = np.maximum(ll0, ll1)
    a = np.exp(ll0 - m)
    b = np.exp(ll1 - m)
    return b / (a + b + 1e-12)


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


def tune_threshold(y_true, prob):
    best = None

    for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = (prob >= thr).astype(np.int64)
        m = compute_metrics(y_true, pred)
        m["threshold"] = float(thr)

        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = m

    return float(best["threshold"]), best


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
            writer.writerow({"model": "gmm_mfcc_vector_torch_gpu_test", "split": split, **metrics[split]})


def main():
    ensure_dirs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Device:", device)

    xtr, ytr, _ = load_npz(TRAIN_FEATS)
    xva, yva, _ = load_npz(VAL_FEATS)
    xte, yte, pte = load_npz(TEST_FEATS)

    xtr, xva, xte, mean, std = standardize(xtr, xva, xte)

    xtr_t = torch.tensor(xtr, dtype=torch.float32, device=device)
    xva_t = torch.tensor(xva, dtype=torch.float32, device=device)
    xte_t = torch.tensor(xte, dtype=torch.float32, device=device)

    print("[INFO] M5 PyTorch GPU GMM over MFCC vectors")
    print("[INFO] Train:", xtr.shape, "Val:", xva.shape, "Test:", xte.shape)

    best = None

    for k in K_VALUES:
        print(f"[INFO] Training K={k}")

        model = {
            0: train_gmm_torch(xtr_t[ytr == 0], k, SEED + k),
            1: train_gmm_torch(xtr_t[ytr == 1], k, SEED + k + 100),
        }

        ll0 = class_log_likelihood_torch(xva_t, model[0], device)
        ll1 = class_log_likelihood_torch(xva_t, model[1], device)

        prob = prob_stutter(ll0, ll1)
        threshold, val_metrics = tune_threshold(yva, prob)

        print(
            f"[K={k}] val_acc={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} thr={threshold:.2f}"
        )

        if best is None or val_metrics["macro_f1"] > best["val"]["macro_f1"]:
            best = {
                "k": k,
                "model": model,
                "threshold": threshold,
                "val": val_metrics,
            }

    model = best["model"]
    threshold = best["threshold"]

    ll0 = class_log_likelihood_torch(xte_t, model[0], device)
    ll1 = class_log_likelihood_torch(xte_t, model[1], device)
    prob = prob_stutter(ll0, ll1)

    pred = (prob >= threshold).astype(np.int64)
    test_metrics = compute_metrics(yte, pred)
    test_metrics["threshold"] = float(threshold)

    with OUT_PREDS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "y_true", "y_pred", "prob_stutter", "ll_fluent", "ll_stutter"])
        for path, yt, yp, ps, a, b in zip(pte, yte, pred, prob, ll0, ll1):
            writer.writerow([path, int(yt), int(yp), float(ps), float(a), float(b)])

    torch.save(
        {
            "model": "gmm_mfcc_vector_torch_gpu_test",
            "best_k_components": int(best["k"]),
            "threshold": float(threshold),
            "mean": torch.tensor(mean),
            "std": torch.tensor(std),
            "fluent": model[0],
            "stutter": model[1],
        },
        OUT_MODEL,
    )

    final = {
        "model": "gmm_mfcc_vector_torch_gpu_test",
        "member_task": "M5 GPU test - PyTorch GMM over MFCC vectors",
        "device": device,
        "features": "MFCC + delta + delta-delta statistics vector",
        "best_k_components": int(best["k"]),
        "threshold": float(threshold),
        "val": best["val"],
        "test": test_metrics,
    }

    OUT_METRICS.write_text(json.dumps(final, indent=2), encoding="utf-8")
    save_metrics_csv({"val": best["val"], "test": test_metrics})

    print("\n[OK] Saved:", OUT_MODEL)
    print("[OK] Saved:", OUT_METRICS)
    print("[OK] Saved:", OUT_PREDS)
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
