from pathlib import Path
import json
import csv

OUT_CSV = Path("outputs/metrics/final_all_models_comparison.csv")
OUT_MD = Path("outputs/metrics/final_all_models_comparison.md")


ROWS = [
    {
        "model": "Diagonal Gaussian Baseline",
        "features": "MFCC + delta + delta-delta statistics",
        "accuracy": 0.5612,
        "macro_f1": 0.5201,
        "stutter_recall": 0.5466,
        "fluent_specificity": 0.6132,
        "stutter_f1": 0.6605,
        "fluent_f1": 0.3797,
        "threshold": 0.50,
    },
    {
        "model": "CNN-BiLSTM-Attention v1",
        "features": "Log-mel spectrogram",
        "accuracy": 0.7066,
        "macro_f1": 0.6159,
        "stutter_recall": 0.7635,
        "fluent_specificity": 0.5038,
        "stutter_f1": 0.8026,
        "fluent_f1": 0.4293,
        "threshold": 0.50,
    },
    {
        "model": "CNN-BiLSTM-Attention v2",
        "features": "Log-mel + balanced sampler + SpecAugment",
        "accuracy": 0.7623,
        "macro_f1": 0.6565,
        "stutter_recall": 0.8434,
        "fluent_specificity": 0.4733,
        "stutter_f1": 0.8472,
        "fluent_f1": 0.4659,
        "threshold": 0.30,
    },
    {
        "model": "CNN-BiLSTM-Attention v2.1",
        "features": "v2 + threshold tuning",
        "accuracy": 0.7515,
        "macro_f1": 0.6673,
        "stutter_recall": 0.8031,
        "fluent_specificity": 0.5674,
        "stutter_f1": 0.8346,
        "fluent_f1": 0.5000,
        "threshold": 0.34,
    },
    {
        "model": "CNN-BiLSTM-MFCC Fusion v3",
        "features": "Log-mel + MFCC fusion + focal loss",
        "accuracy": 0.7334,
        "macro_f1": 0.6601,
        "stutter_recall": 0.7667,
        "fluent_specificity": 0.6145,
        "stutter_f1": 0.8179,
        "fluent_f1": 0.5023,
        "threshold": 0.46,
    },
]


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(ROWS)

    lines = []
    lines.append("# Final All-Model Comparison")
    lines.append("")
    lines.append("| Model | Features | Accuracy | Macro-F1 | Stutter Recall | Fluent Specificity | Stutter F1 | Fluent F1 | Threshold |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in ROWS:
        lines.append(
            f"| {row['model']} "
            f"| {row['features']} "
            f"| {pct(row['accuracy'])} "
            f"| {pct(row['macro_f1'])} "
            f"| {pct(row['stutter_recall'])} "
            f"| {pct(row['fluent_specificity'])} "
            f"| {pct(row['stutter_f1'])} "
            f"| {pct(row['fluent_f1'])} "
            f"| {row['threshold']:.2f} |"
        )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] Saved:", OUT_CSV)
    print("[OK] Saved:", OUT_MD)
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()