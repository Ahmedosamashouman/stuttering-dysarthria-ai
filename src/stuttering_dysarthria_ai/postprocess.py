from __future__ import annotations

from typing import Dict, Tuple


def probabilities_to_prediction(
    probabilities: Dict[str, float],
    threshold: float,
    positive_label: str = "stutter",
) -> Tuple[str, float]:
    stutter_score = float(probabilities.get(positive_label, 0.0))

    if stutter_score >= threshold:
        return positive_label, stutter_score

    fluent_score = float(probabilities.get("fluent", 1.0 - stutter_score))
    return "fluent", fluent_score
