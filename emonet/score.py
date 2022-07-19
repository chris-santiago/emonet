"""
Console script for scoring.
"""
import argparse

from emonet import ROOT
from emonet.model import EmotionRegressor


def get_saved(emotion: str):
    """Get saved model."""
    return ROOT.joinpath("saved_models", f"{emotion}.ckpt")


def get_parser():
    """Get an ArgumentParser instance."""
    parser = argparse.ArgumentParser(
        description="Score an audio file on a specific emotion."
    )
    parser.add_argument(
        "emotion",
        type=str,
        help="Emotion model to score file.",
        choices=["anger", "fear", "sadness"],
    )
    parser.add_argument("file", type=str, help="Filename to score.")
    return parser


def main():
    """Main scoring function."""
    parser = get_parser()
    args = parser.parse_args()
    model = EmotionRegressor.load_from_checkpoint(get_saved(args.emotion))
    score = round(model.score_file(args.file).item(), 2)
    return score


if __name__ == "__main__":
    print(main())
