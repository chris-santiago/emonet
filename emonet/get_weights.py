"""
Module to calculate emotion weights per therapist and export to JSON file.

Output can be read into loss functions for weighting during classification.
"""
from collections import Counter
from typing import Dict

from emonet import DATA_DIR, EMOTIONS, THERAPISTS
from emonet.utils import from_json, to_json


def get_weights(
    therapist: str, emotion: str, split: str = "train", binary: bool = True
) -> Dict:
    """
    Get therapist emotion rates in order of increasing severity.

    Parameters
    ----------
    therapist: str
        Therapist weights to retrieve.
    emotion: str
        Emotion to retrieve.
    split: str
        Split to retreive
    binary: bool
        Indicator to retrieve binary yes/no labels or original none/low/medium/high labels.

    Returns
    -------
    Dict
        Dictionary record including therapist, emotion, counts and weights per.
    """
    dataset = from_json(DATA_DIR.joinpath(therapist, f"{split}_splits.json"))
    c: Counter = Counter()
    c.update([itm[emotion] for itm in dataset.values()])
    if binary:
        counts = [
            c.get("none", 0) + c.get("low", 0),
            c.get("medium", 0) + c.get("high", 0),
        ]
    else:
        counts = [
            c.get("none", 0),
            c.get("low", 0),
            c.get("medium", 0),
            c.get("high", 0),
        ]
    weights = [max(counts) / x for x in counts]
    return {
        "therapist": therapist,
        "emotion": emotion,
        "counts": counts,
        "weights": weights,
    }


if __name__ == "__main__":
    weights = []
    for therapist in THERAPISTS:
        for emotion in EMOTIONS:
            weights.append(get_weights(therapist, emotion, "valid", binary=True))
    to_json(weights, DATA_DIR.joinpath("therapist-emotion-weights-valid.json"))
