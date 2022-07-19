"""
Training module.

To run via CLI:

`python train.py <num_workers> <max_epochs> <emotion>`

Examples
--------
`python train.py 12 300 anger`
"""
import pathlib
import sys

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from emonet import DATA_DIR
from emonet.data_loader import TQRegressionDataset
from emonet.model import EmotionRegressor

PAPERSPACE_DATA_DIR = pathlib.Path("/datasets/emonet-data/vad")


def main(
    workers: int = 1,
    epochs: int = 1,
    emotion: str = "sadness",
    use_wandb: bool = False,
    data_dir: pathlib.Path = DATA_DIR,
):
    """
    Train model.

    Parameters
    ----------
    workers: int
        Number of workers to use.
    epochs: int
        Number of epochs to train.
    emotion: str
        Emotion to train model on.
    use_wandb: bool
        Whether to log results using Weights & Biases
    data_dir: pathlib.Path
        Path to directory containing training/validation data and manifests.

    Returns
    -------
    Tuple
        Model and trainer objects.
    """
    data = {
        "train": TQRegressionDataset(
            data_dir.joinpath("train_splits.json"), data_dir, emotion=emotion
        ),
        "valid": TQRegressionDataset(
            data_dir.joinpath("valid_splits.json"), data_dir, emotion=emotion
        ),
        "test": TQRegressionDataset(
            data_dir.joinpath("test_splits.json"), data_dir, emotion=emotion
        ),
    }

    train_dl = DataLoader(data["train"], 64, num_workers=workers, shuffle=True)
    valid_dl = DataLoader(data["valid"], 64, num_workers=workers, shuffle=False)

    model = EmotionRegressor(
        emotion=emotion,
        lr=0.001,
        freq_mask=30,
        time_mask=30,
        weight_decay=0.01,
        n_fft=641,
    )

    if use_wandb:
        wandb_logger = WandbLogger(
            project="emonet-regressor",
            name=emotion,
            log_model="all",
            tags=[
                "regressor",
                emotion,
                "maxPool",
                "fc2",
                "adamW",
                "gru",
                "bidirectional",
                "vad",
            ],
            notes=f"Training {emotion} regressor on VAD 8sec chunks.",
        )
        trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger, accelerator="cpu")
    else:
        trainer = pl.Trainer(max_epochs=epochs, accelerator="cpu")
    trainer.fit(model, train_dl, valid_dl)
    return model, trainer


if __name__ == "__main__":
    defaults = {"workers": 12, "epochs": 300, "emotion": "sadness"}
    for i, key in enumerate(defaults.keys(), start=1):
        try:
            defaults[key] = sys.argv[i]  # if running via CLI
        except IndexError:
            pass  # if running via IDE
    main(int(defaults["workers"]), int(defaults["epochs"]), defaults["emotion"])
