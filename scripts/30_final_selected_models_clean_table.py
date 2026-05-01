from __future__ import annotations

import json
from pathlib import Path


OUT_MD = Path("outputs/metrics/final_selected_models_clean_table.md")


def pct(x):
    return f"{x * 100:.2f}%"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_test(path):
    m = read(path)
    return m["test"]


def main():
    rows = []

    m5 = get_test("outputs/metrics/gmm_mfcc_vector_metrics.json")
    rows.append(["M5", "True GMM over MFCC Vector", m5])

    m6 = get_test("outputs/metrics/gmm_hmm_mfcc_sequence_metrics.json")
    rows.append(["M6", "GMM-HMM over MFCC Sequence", m6])

    # Keep these fixed from your already confirmed final results.
    fixed = [
        ["M7-1", "CNN-BiLSTM + MFCC Fusion v3", "76.26%", "65.98%", "84.02%", "48.60%", "Full test"],
        ["M7-2", "CNN-BiLSTM v2 Balanced", "76.23%", "65.65%", "84.34%", "47.33%", "Full test"],
        ["M8-1", "Wav2Vec2 Full Attention", "76.21%", "76.15%", "71.37%", "81.04%", "Balanced 1572 samples"],
        ["M8-2", "Wav2Vec2 SSL Bigger", "75.89%", "75.85%", "71.76%", "80.03%", "Balanced 1572 samples"],
        ["M9", "CNN + Wav2Vec2 Forced Ensemble", "77.25%", "77.00%", "66.75%", "87.75%", "800 common samples"],
    ]

    lines = []
    lines.append("# Final Selected Models Clean Table")
    lines.append("")
    lines.append("| Task | Model | Accuracy | Macro-F1 | Stutter Recall | Fluent Specificity | Test setup |")
    lines.append("|---|---|---:|---:|---:|---:|---|")

    for task, name, m in rows:
        lines.append(
            f"| {task} | {name} | {pct(m['accuracy'])} | {pct(m['macro_f1'])} | "
            f"{pct(m['recall_stutter_sensitivity'])} | {pct(m['specificity_fluent'])} | Full test |"
        )

    for r in fixed:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- M5 and M6 are now friend-style classical baselines.")
    lines.append("- Old diagonal Gaussian M5 and simple Gaussian HMM M6 were archived.")
    lines.append("- M9 is the final ensemble result, not a separate training model.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
