from pathlib import Path
import json
import csv

METRIC_FILES = [
    {
        "name": "Diagonal Gaussian Baseline",
        "features": "MFCC + delta + delta-delta statistics",
        "path": Path("outputs/metrics/baseline_diag_gaussian_metrics.json"),
    },
    {
        "name": "CNN-BiLSTM-Attention v1",
        "features": "Log-mel spectrogram",
        "path": Path("outputs/metrics/cnn_bilstm_attention_metrics.json"),
    },
    {
        "name": "CNN-BiLSTM-Attention v2 Balanced",
        "features": "Log-mel + balanced sampler + SpecAugment + threshold tuning",
        "path": Path("outputs/metrics/cnn_bilstm_attention_v2_metrics.json"),
    },
]

OUT_CSV = Path("outputs/metrics/final_model_comparison.csv")
OUT_MD = Path("outputs/metrics/final_model_comparison.md")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def row_from_metrics(name, features, metrics):
    test = metrics["test"]

    return {
        "model": name,
        "features": features,
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "precision_stutter": test["precision_stutter"],
        "recall_stutter_sensitivity": test["recall_stutter_sensitivity"],
        "specificity_fluent": test["specificity_fluent"],
        "f1_stutter": test["f1_stutter"],
        "f1_fluent": test["f1_fluent"],
        "threshold": test.get("threshold", 0.5),
        "tp": test["tp"],
        "tn": test["tn"],
        "fp": test["fp"],
        "fn": test["fn"],
    }


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    rows = []

    for item in METRIC_FILES:
        if item["path"].exists():
            metrics = load_json(item["path"])
            rows.append(row_from_metrics(item["name"], item["features"], metrics))
        else:
            print(f"[SKIP] missing: {item['path']}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    lines.append("# Final Model Comparison")
    lines.append("")
    lines.append("| Model | Features | Accuracy | Macro-F1 | Stutter Recall | Fluent Specificity | Stutter F1 | Fluent F1 | Threshold |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            f"| {row['model']} "
            f"| {row['features']} "
            f"| {pct(row['accuracy'])} "
            f"| {pct(row['macro_f1'])} "
            f"| {pct(row['recall_stutter_sensitivity'])} "
            f"| {pct(row['specificity_fluent'])} "
            f"| {pct(row['f1_stutter'])} "
            f"| {pct(row['f1_fluent'])} "
            f"| {row['threshold']:.2f} |"
        )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Saved:", OUT_CSV)
    print("[OK] Saved:", OUT_MD)
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()