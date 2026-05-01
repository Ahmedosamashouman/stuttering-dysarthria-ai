from __future__ import annotations

import csv
import json
from pathlib import Path

from stuttering_dysarthria_ai.inference import SpeechPathologyPredictor


MANIFEST = Path("data/processed/manifest.csv")


def pick_sample(label: str) -> str:
    with MANIFEST.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test" and row["label"] == label:
                return row["path"]

    raise RuntimeError(f"No test sample found for label: {label}")


def main():
    predictor = SpeechPathologyPredictor("outputs/production_model")

    for label in ["fluent", "stutter"]:
        path = pick_sample(label)
        result = predictor.predict_from_path(path)

        print("\n" + "=" * 70)
        print("Ground truth:", label)
        print("Path:", path)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
