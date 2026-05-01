from __future__ import annotations


ALPHA_SSL = 0.47
ALPHA_CNN = 0.53

LOW_THR = 0.22
HIGH_THR = 0.58


def high_confidence_decision(
    cnn_prob_stutter: float,
    ssl_prob_stutter: float,
) -> dict:
    """
    High-confidence screening rule.

    If combined probability is low enough, predict fluent.
    If combined probability is high enough, predict stutter.
    Otherwise, return uncertain instead of forcing a wrong prediction.
    """

    combined_prob = (ALPHA_SSL * ssl_prob_stutter) + (ALPHA_CNN * cnn_prob_stutter)

    if combined_prob <= LOW_THR:
        return {
            "prediction": "fluent",
            "decision": "accepted",
            "confidence_mode": "high_confidence",
            "combined_probability_stutter": float(combined_prob),
            "low_threshold": LOW_THR,
            "high_threshold": HIGH_THR,
            "alpha_ssl": ALPHA_SSL,
            "alpha_cnn": ALPHA_CNN,
            "message": "High-confidence fluent prediction.",
        }

    if combined_prob >= HIGH_THR:
        return {
            "prediction": "stutter",
            "decision": "accepted",
            "confidence_mode": "high_confidence",
            "combined_probability_stutter": float(combined_prob),
            "low_threshold": LOW_THR,
            "high_threshold": HIGH_THR,
            "alpha_ssl": ALPHA_SSL,
            "alpha_cnn": ALPHA_CNN,
            "message": "High-confidence stutter prediction.",
        }

    return {
        "prediction": "uncertain",
        "decision": "rejected",
        "confidence_mode": "low_confidence",
        "combined_probability_stutter": float(combined_prob),
        "low_threshold": LOW_THR,
        "high_threshold": HIGH_THR,
        "alpha_ssl": ALPHA_SSL,
        "alpha_cnn": ALPHA_CNN,
        "message": "Uncertain result. Please record a longer or clearer speech sample.",
    }
