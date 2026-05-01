from pathlib import Path
import csv
import json

import numpy as np

FEATURE_DIR = Path("data/processed/features")
OUT_MODEL = Path("outputs/models/baseline_diag_gaussian.npz")
OUT_METRICS = Path("outputs/metrics/baseline_diag_gaussian_metrics.json")
OUT_PREDS = Path("outputs/predictions/baseline_diag_gaussian_test_predictions.csv")


def load_split(split):
    data = np.load(FEATURE_DIR / f"{split}_features.npz")
    x = data["mfcc_stats"].astype(np.float32)
    y = data["labels"].astype(np.int64)
    paths = data["paths"]
    return x, y, paths


def standardize_train_val_test(x_train, x_val, x_test):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-8

    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return x_train, x_val, x_test, mean, std


def fit_diag_gaussian(x, y):
    classes = sorted(np.unique(y).tolist())

    means = {}
    vars_ = {}
    priors = {}

    for c in classes:
        xc = x[y == c]
        means[c] = xc.mean(axis=0)
        vars_[c] = xc.var(axis=0) + 1e-6
        priors[c] = len(xc) / len(x)

    return classes, means, vars_, priors


def log_gaussian_diag(x, mean, var):
    # x shape: samples × features
    # mean/var shape: features
    log_det = np.sum(np.log(var))
    quad = np.sum(((x - mean) ** 2) / var, axis=1)
    d = x.shape[1]
    return -0.5 * (d * np.log(2 * np.pi) + log_det + quad)


def predict_diag_gaussian(x, classes, means, vars_, priors):
    score_list = []

    for c in classes:
        score = log_gaussian_diag(x, means[c], vars_[c]) + np.log(priors[c] + 1e-12)
        score_list.append(score)

    scores = np.vstack(score_list).T
    pred_index = np.argmax(scores, axis=1)
    preds = np.array([classes[i] for i in pred_index], dtype=np.int64)

    # Convert scores to probability-like value for stutter class.
    # Binary case: class 1 score against class 0 score.
    if 0 in classes and 1 in classes:
        s0 = scores[:, classes.index(0)]
        s1 = scores[:, classes.index(1)]
        prob_stutter = 1.0 / (1.0 + np.exp(-(s1 - s0)))
    else:
        prob_stutter = scores.max(axis=1)

    return preds, prob_stutter, scores


def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(1, len(y_true))

    precision_1 = tp / max(1, tp + fp)
    recall_1 = tp / max(1, tp + fn)
    f1_1 = 2 * precision_1 * recall_1 / max(1e-12, precision_1 + recall_1)

    precision_0 = tn / max(1, tn + fn)
    recall_0 = tn / max(1, tn + fp)
    f1_0 = 2 * precision_0 * recall_0 / max(1e-12, precision_0 + recall_0)

    macro_f1 = (f1_0 + f1_1) / 2
    specificity = recall_0

    return {
        "accuracy": accuracy,
        "precision_stutter": precision_1,
        "recall_stutter_sensitivity": recall_1,
        "specificity_fluent": specificity,
        "f1_stutter": f1_1,
        "f1_fluent": f1_0,
        "macro_f1": macro_f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def save_predictions(paths, y_true, y_pred, prob_stutter):
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PREDS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "y_true", "y_pred", "prob_stutter"])

        for p, yt, yp, ps in zip(paths, y_true, y_pred, prob_stutter):
            writer.writerow([p, int(yt), int(yp), float(ps)])


def main():
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREDS.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading features...")
    x_train, y_train, _ = load_split("train")
    x_val, y_val, _ = load_split("val")
    x_test, y_test, test_paths = load_split("test")

    print("[INFO] Shapes:")
    print("x_train:", x_train.shape, "y_train:", y_train.shape)
    print("x_val:", x_val.shape, "y_val:", y_val.shape)
    print("x_test:", x_test.shape, "y_test:", y_test.shape)

    print("[INFO] Standardizing features...")
    x_train, x_val, x_test, scaler_mean, scaler_std = standardize_train_val_test(
        x_train, x_val, x_test
    )

    print("[INFO] Training diagonal Gaussian baseline...")
    classes, means, vars_, priors = fit_diag_gaussian(x_train, y_train)

    print("[INFO] Evaluating validation...")
    val_pred, val_prob, _ = predict_diag_gaussian(x_val, classes, means, vars_, priors)
    val_metrics = compute_metrics(y_val, val_pred)

    print("[INFO] Evaluating test...")
    test_pred, test_prob, _ = predict_diag_gaussian(x_test, classes, means, vars_, priors)
    test_metrics = compute_metrics(y_test, test_pred)

    all_metrics = {
        "model": "baseline_diag_gaussian_mfcc_stats",
        "features": "MFCC + delta + delta-delta statistics",
        "val": val_metrics,
        "test": test_metrics,
    }

    np.savez_compressed(
        OUT_MODEL,
        classes=np.array(classes),
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        mean_0=means[0],
        var_0=vars_[0],
        prior_0=priors[0],
        mean_1=means[1],
        var_1=vars_[1],
        prior_1=priors[1],
    )

    with OUT_METRICS.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    save_predictions(test_paths, y_test, test_pred, test_prob)

    print("\n[OK] Saved model:", OUT_MODEL)
    print("[OK] Saved metrics:", OUT_METRICS)
    print("[OK] Saved predictions:", OUT_PREDS)

    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()