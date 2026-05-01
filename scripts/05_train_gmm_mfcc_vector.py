from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


TRAIN_FEATS = Path("data/processed/features/train_features.npz")
VAL_FEATS = Path("data/processed/features/val_features.npz")
TEST_FEATS = Path("data/processed/features/test_features.npz")

OUT_MODEL = Path("outputs/models/gmm_mfcc_vector.npz")
OUT_METRICS_JSON = Path("outputs/metrics/gmm_mfcc_vector_metrics.json")
OUT_METRICS_CSV = Path("outputs/metrics/gmm_mfcc_vector_metrics.csv")
OUT_PREDS = Path("outputs/predictions/gmm_mfcc_vector_test_predictions.csv")

K_VALUES = [1, 2, 4, 8]
MAX_EM_ITER = 35
SEED = 42
EPS = 1e-6


def ensure_dirs():
    for p in [OUT_MODEL, OUT_METRICS_JSON, OUT_METRICS_CSV, OUT_PREDS]:
        p.parent.mkdir(parents=True, exist_ok=True)


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    if "mfcc_stats" not in keys or "labels" not in keys:
        raise ValueError(f"{path} must contain mfcc_stats and labels. Found: {data.files}")

    X = data["mfcc_stats"].astype(np.float32)
    y = data["labels"].astype(np.int64)

    if "paths" in keys:
        paths = data["paths"]
    else:
        paths = np.array([f"sample_{i}" for i in range(len(y))], dtype=object)

    paths = np.asarray([str(p) for p in paths], dtype=object)
    return X, y, paths


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std, mean, std


def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-12)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def diag_gaussian_logpdf(X, means, vars_):
    # X: [N, D], means/vars: [K, D] -> [N, K]
    X2 = X[:, None, :]
    diff = X2 - means[None, :, :]
    log_det = np.sum(np.log(2.0 * np.pi * vars_), axis=1)
    maha = np.sum((diff ** 2) / vars_[None, :, :], axis=2)
    return -0.5 * (log_det[None, :] + maha)


def init_gmm(X, k, rng):
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=n < k)
    means = X[idx].copy()
    vars_ = np.tile(np.var(X, axis=0, keepdims=True) + 1e-3, (k, 1))
    weights = np.ones(k, dtype=np.float64) / k
    return weights, means, vars_


def train_gmm(X, k, rng):
    X = X.astype(np.float64)
    n, d = X.shape
    weights, means, vars_ = init_gmm(X, k, rng)

    prev_ll = None

    for _ in range(MAX_EM_ITER):
        log_probs = diag_gaussian_logpdf(X, means, vars_) + np.log(weights[None, :] + 1e-12)
        log_norm = logsumexp(log_probs, axis=1, keepdims=True)
        resp = np.exp(log_probs - log_norm)

        Nk = resp.sum(axis=0) + 1e-8
        weights = Nk / n
        means = (resp.T @ X) / Nk[:, None]

        for j in range(k):
            diff = X - means[j]
            vars_[j] = (resp[:, j:j+1] * diff * diff).sum(axis=0) / Nk[j]

        vars_ = np.maximum(vars_, 1e-4)

        ll = float(log_norm.sum() / n)
        if prev_ll is not None and abs(ll - prev_ll) < 1e-4:
            break
        prev_ll = ll

    return {
        "weights": weights.astype(np.float64),
        "means": means.astype(np.float64),
        "vars": vars_.astype(np.float64),
    }


def class_log_likelihood(X, gmm):
    log_probs = diag_gaussian_logpdf(X, gmm["means"], gmm["vars"]) + np.log(gmm["weights"][None, :] + 1e-12)
    return logsumexp(log_probs, axis=1)


def prob_stutter_from_ll(ll_fluent, ll_stutter):
    m = np.maximum(ll_fluent, ll_stutter)
    ef = np.exp(ll_fluent - m)
    es = np.exp(ll_stutter - m)
    return es / (ef + es + 1e-12)


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


def tune_threshold(y_true, prob_stutter):
    best = None
    rows = []

    for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = (prob_stutter >= thr).astype(np.int64)
        m = compute_metrics(y_true, pred)
        m["threshold"] = float(thr)
        rows.append(m)

        if best is None or m["macro_f1"] > best["macro_f1"]:
            best = m

    return float(best["threshold"]), best, rows


def predict_probs(X, model):
    ll0 = class_log_likelihood(X, model[0])
    ll1 = class_log_likelihood(X, model[1])
    return prob_stutter_from_ll(ll0, ll1), ll0, ll1


def save_metrics_csv(metrics):
    with OUT_METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model", "split", "accuracy", "macro_f1", "balanced_accuracy",
            "precision_stutter", "recall_stutter_sensitivity", "specificity_fluent",
            "f1_stutter", "f1_fluent", "tp", "tn", "fp", "fn", "threshold"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split in ["val", "test"]:
            row = {"model": "gmm_mfcc_vector", "split": split, **metrics[split]}
            writer.writerow(row)


def main():
    ensure_dirs()
    rng = np.random.default_rng(SEED)

    Xtr, ytr, _ = load_npz(TRAIN_FEATS)
    Xva, yva, _ = load_npz(VAL_FEATS)
    Xte, yte, pte = load_npz(TEST_FEATS)

    Xtr, Xva, Xte, mean, std = standardize(Xtr, Xva, Xte)

    print("[INFO] M5 True GMM over MFCC vectors")
    print("[INFO] Train:", Xtr.shape, "Val:", Xva.shape, "Test:", Xte.shape)

    best_pack = None

    for k in K_VALUES:
        print(f"[INFO] Training class GMMs with K={k}")
        model = {
            0: train_gmm(Xtr[ytr == 0], k, rng),
            1: train_gmm(Xtr[ytr == 1], k, rng),
        }

        va_prob, _, _ = predict_probs(Xva, model)
        threshold, val_metrics, _ = tune_threshold(yva, va_prob)

        print(f"[K={k}] val_macro_f1={val_metrics['macro_f1']:.4f} val_acc={val_metrics['accuracy']:.4f} thr={threshold:.2f}")

        if best_pack is None or val_metrics["macro_f1"] > best_pack["val"]["macro_f1"]:
            best_pack = {
                "k": k,
                "model": model,
                "threshold": threshold,
                "val": val_metrics,
            }

    model = best_pack["model"]
    threshold = best_pack["threshold"]
    k = best_pack["k"]

    te_prob, ll0, ll1 = predict_probs(Xte, model)
    te_pred = (te_prob >= threshold).astype(np.int64)
    test_metrics = compute_metrics(yte, te_pred)
    test_metrics["threshold"] = threshold

    val_metrics = best_pack["val"]

    with OUT_PREDS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "y_true", "y_pred", "prob_stutter", "ll_fluent", "ll_stutter"])
        for p, yt, yp, ps, a, b in zip(pte, yte, te_pred, te_prob, ll0, ll1):
            writer.writerow([p, int(yt), int(yp), float(ps), float(a), float(b)])

    np.savez(
        OUT_MODEL,
        k=np.array(k),
        threshold=np.array(threshold),
        mean=mean,
        std=std,
        fluent_weights=model[0]["weights"],
        fluent_means=model[0]["means"],
        fluent_vars=model[0]["vars"],
        stutter_weights=model[1]["weights"],
        stutter_means=model[1]["means"],
        stutter_vars=model[1]["vars"],
    )

    final = {
        "model": "gmm_mfcc_vector",
        "member_task": "M5 Classical Baseline 1 - True GMM over MFCC vectors",
        "features": "MFCC + delta + delta-delta statistics vector",
        "best_k_components": int(k),
        "threshold": float(threshold),
        "val": val_metrics,
        "test": test_metrics,
    }

    OUT_METRICS_JSON.write_text(json.dumps(final, indent=2), encoding="utf-8")
    save_metrics_csv({"val": val_metrics, "test": test_metrics})

    print("\n[OK] Saved model:", OUT_MODEL)
    print("[OK] Saved metrics:", OUT_METRICS_JSON)
    print("[OK] Saved predictions:", OUT_PREDS)
    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
