from pathlib import Path
import csv

CNN_PREDS = Path("outputs/predictions/cnn_bilstm_attention_v2_test_predictions.csv")
SSL_PREDS = Path("outputs/predictions/wav2vec2_ssl_subset_test_predictions.csv")

OUT_CSV = Path("outputs/metrics/ssl_cnn_ensemble_search.csv")
OUT_MD = Path("outputs/metrics/ssl_cnn_ensemble_search.md")


def load_preds(path):
    data = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            data[row["path"]] = {
                "y_true": int(row["y_true"]),
                "prob_stutter": float(row["prob_stutter"]),
            }

    return data


def compute_metrics(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / max(1, len(y_true))

    precision_stutter = tp / max(1, tp + fp)
    recall_stutter = tp / max(1, tp + fn)
    specificity_fluent = tn / max(1, tn + fp)

    f1_stutter = 2 * precision_stutter * recall_stutter / max(
        1e-12, precision_stutter + recall_stutter
    )

    precision_fluent = tn / max(1, tn + fn)
    f1_fluent = 2 * precision_fluent * specificity_fluent / max(
        1e-12, precision_fluent + specificity_fluent
    )

    macro_f1 = (f1_stutter + f1_fluent) / 2
    balanced_accuracy = (recall_stutter + specificity_fluent) / 2

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "precision_stutter": precision_stutter,
        "recall_stutter_sensitivity": recall_stutter,
        "specificity_fluent": specificity_fluent,
        "f1_stutter": f1_stutter,
        "f1_fluent": f1_fluent,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def pct(x):
    return f"{x * 100:.2f}%"


def evaluate_probs(name, y_true, probs, extra):
    rows = []

    for i in range(5, 96):
        threshold = i / 100
        y_pred = [1 if p >= threshold else 0 for p in probs]
        metrics = compute_metrics(y_true, y_pred)

        rows.append({
            "method": name,
            "threshold": threshold,
            **extra,
            **metrics,
        })

    return rows


def main():
    if not CNN_PREDS.exists():
        raise FileNotFoundError(CNN_PREDS)

    if not SSL_PREDS.exists():
        raise FileNotFoundError(SSL_PREDS)

    cnn = load_preds(CNN_PREDS)
    ssl = load_preds(SSL_PREDS)

    common_paths = sorted(set(cnn.keys()) & set(ssl.keys()))

    if not common_paths:
        raise RuntimeError("No common paths between CNN and SSL prediction files.")

    y_true = [ssl[p]["y_true"] for p in common_paths]
    cnn_probs = [cnn[p]["prob_stutter"] for p in common_paths]
    ssl_probs = [ssl[p]["prob_stutter"] for p in common_paths]

    all_rows = []

    all_rows.extend(
        evaluate_probs(
            "CNN v2 on SSL subset",
            y_true,
            cnn_probs,
            {"alpha_ssl": 0.0, "alpha_cnn": 1.0},
        )
    )

    all_rows.extend(
        evaluate_probs(
            "Wav2Vec2 SSL",
            y_true,
            ssl_probs,
            {"alpha_ssl": 1.0, "alpha_cnn": 0.0},
        )
    )

    # Ensemble search:
    # combined = alpha_ssl * ssl_prob + alpha_cnn * cnn_prob
    for a in range(0, 101):
        alpha_ssl = a / 100
        alpha_cnn = 1.0 - alpha_ssl

        combined = [
            alpha_ssl * s + alpha_cnn * c
            for s, c in zip(ssl_probs, cnn_probs)
        ]

        all_rows.extend(
            evaluate_probs(
                "CNN + Wav2Vec2 ensemble",
                y_true,
                combined,
                {
                    "alpha_ssl": alpha_ssl,
                    "alpha_cnn": alpha_cnn,
                },
            )
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "alpha_ssl",
        "alpha_cnn",
        "threshold",
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

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    best_acc = max(all_rows, key=lambda r: r["accuracy"])
    best_macro = max(all_rows, key=lambda r: r["macro_f1"])
    best_balanced = max(all_rows, key=lambda r: r["balanced_accuracy"])

    selected = [
        ("Best Accuracy", best_acc),
        ("Best Macro-F1", best_macro),
        ("Best Balanced Accuracy", best_balanced),
    ]

    lines = []
    lines.append("# SSL + CNN Threshold / Ensemble Search")
    lines.append("")
    lines.append(f"Common test samples: {len(common_paths)}")
    lines.append("")
    lines.append("| Selection | Method | Alpha SSL | Alpha CNN | Threshold | Accuracy | Macro-F1 | Balanced Acc | Stutter Recall | Fluent Specificity | Stutter F1 | Fluent F1 | FP | FN |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for label, row in selected:
        lines.append(
            f"| {label} "
            f"| {row['method']} "
            f"| {row['alpha_ssl']:.2f} "
            f"| {row['alpha_cnn']:.2f} "
            f"| {row['threshold']:.2f} "
            f"| {pct(row['accuracy'])} "
            f"| {pct(row['macro_f1'])} "
            f"| {pct(row['balanced_accuracy'])} "
            f"| {pct(row['recall_stutter_sensitivity'])} "
            f"| {pct(row['specificity_fluent'])} "
            f"| {pct(row['f1_stutter'])} "
            f"| {pct(row['f1_fluent'])} "
            f"| {row['fp']} "
            f"| {row['fn']} |"
        )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Saved:", OUT_CSV)
    print("[OK] Saved:", OUT_MD)
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()