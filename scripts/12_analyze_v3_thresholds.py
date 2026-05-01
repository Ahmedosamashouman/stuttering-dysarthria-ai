from pathlib import Path
import csv

PRED_FILE = Path("outputs/predictions/cnn_bilstm_mfcc_fusion_v3_test_predictions.csv")
OUT_CSV = Path("outputs/metrics/cnn_bilstm_mfcc_fusion_v3_threshold_analysis.csv")
OUT_MD = Path("outputs/metrics/cnn_bilstm_mfcc_fusion_v3_threshold_analysis.md")


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
    youden_j = recall_stutter + specificity_fluent - 1

    return {
        "accuracy": accuracy,
        "precision_stutter": precision_stutter,
        "recall_stutter_sensitivity": recall_stutter,
        "specificity_fluent": specificity_fluent,
        "f1_stutter": f1_stutter,
        "f1_fluent": f1_fluent,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "youden_j": youden_j,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    y_true = []
    prob_stutter = []

    with PRED_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(row["y_true"]))
            prob_stutter.append(float(row["prob_stutter"]))

    rows = []

    for i in range(5, 96):
        threshold = i / 100
        y_pred = [1 if p >= threshold else 0 for p in prob_stutter]
        metrics = compute_metrics(y_true, y_pred)
        rows.append({"threshold": threshold, **metrics})

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_macro = max(rows, key=lambda r: r["macro_f1"])
    best_j = max(rows, key=lambda r: r["youden_j"])
    candidates_recall_75 = [r for r in rows if r["recall_stutter_sensitivity"] >= 0.75]
    best_specificity_recall_75 = max(candidates_recall_75, key=lambda r: r["specificity_fluent"])

    selected = [
        ("V3 saved threshold", min(rows, key=lambda r: abs(r["threshold"] - 0.44))),
        ("Best Macro-F1", best_macro),
        ("Best Youden J", best_j),
        ("Best specificity with stutter recall >= 75%", best_specificity_recall_75),
    ]

    lines = []
    lines.append("# CNN-BiLSTM MFCC Fusion v3 Threshold Analysis")
    lines.append("")
    lines.append("| Selection | Threshold | Accuracy | Macro-F1 | Stutter Recall | Fluent Specificity | Stutter F1 | Fluent F1 | FP | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for name, row in selected:
        lines.append(
            f"| {name} "
            f"| {row['threshold']:.2f} "
            f"| {pct(row['accuracy'])} "
            f"| {pct(row['macro_f1'])} "
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