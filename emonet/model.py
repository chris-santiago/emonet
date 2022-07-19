"""Emotion model module."""
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchaudio.transforms import FrequencyMasking, MelSpectrogram, TimeMasking
from torchmetrics.functional import mean_absolute_error, mean_absolute_percentage_error

from emonet.modules import (
    ChunkSpectrogram,
    GRUOutput,
    ReshapeOutput,
    TimeDistributed,
    VadChunk,
    WhiteNoise,
    get_vad,
)
from emonet.utils import get_sample
from emonet.wav_splitter import split_sample


class ConvTimeBlock(nn.Module):
    """Convolutional time block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_kernel_size: int = 4,
        pool_stride: int = 4,
        dropout: float = 0.3,
    ):
        """
        Create a time-distributed convolutional time block.

        Parameters
        ----------
        in_channels: int
            Number of input channels for convolutional layer.
        out_channels: int
            Number of output channels for convolutional layer.
        kernel_size: int
            Kernel size for convolutional filter.
        stride: int
            Stride size for convolutional filter.
        padding: int
            Padding size for convolutional filter.
        pool_kernel_size: int
            Kernel size for pooling layer.
        pool_stride: int
            Stride size for pooling layer.
        dropout: float
            Dropout probability.
        """
        super().__init__()

        self.block = nn.Sequential(
            TimeDistributed(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            ),
            TimeDistributed(
                nn.BatchNorm2d(num_features=out_channels),
            ),
            TimeDistributed(
                nn.ReLU(),
            ),
            TimeDistributed(
                nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            ),
            TimeDistributed(nn.Dropout(p=dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through time-distributed convolutional block layer."""
        return self.block(x)


class EmotionRegressor(pl.LightningModule):
    """Emotion regressor model."""

    def __init__(
        self,
        emotion: str,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        snr_low: int = 15,
        snr_high: int = 30,
        freq_mask: int = 20,
        time_mask: int = 20,
        drop: float = 0.3,
        n_mels: int = 128,
        n_fft: int = 128 * 6,
        n_chunks: int = 10,
    ):
        """

        Parameters
        ----------
        emotion: str
            Emotion to train/score.
        lr: float
            Learning rate.
        weight_decay: float
            Weight decay for optimizer.
        snr_low: int
            Signal-to-noise ratio floor for additive white noise augmentation.
        snr_high: int
            Signal-to-noise ratio ceiling for additive white noise augmentation.
        freq_mask: int
            Size of frequency mask for data augmentation.
        time_mask: int
            Size of time mask for data augmentation.
        drop: float
            Dropout rate.
        n_mels: int
            Number of mel filterbanks.
        n_fft: int
            Size of FFT, creates n_fft // 2 + 1 bins.
        n_chunks: int
            Number of time-chunks to create from each sample.
        """
        super().__init__()
        self.emotion = emotion
        self.lr = lr
        self.weight_decay = weight_decay
        self.snr_low = snr_low
        self.snr_high = snr_high
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.drop = drop
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_chunks = n_chunks
        self.save_hyperparameters()
        self.loss_func = nn.MSELoss()

        self.model = nn.Sequential(
            self.conv_block(), self.lstm_block(), self.linear_block()
        )

    def conv_block(self):
        """Time-distributed CNN block."""
        return nn.Sequential(
            ChunkSpectrogram(self.n_chunks),
            ConvTimeBlock(
                in_channels=1, out_channels=16, pool_kernel_size=2, pool_stride=2
            ),
            ConvTimeBlock(
                in_channels=16, out_channels=32, pool_kernel_size=2, pool_stride=2
            ),
            ConvTimeBlock(
                in_channels=32, out_channels=64, pool_kernel_size=2, pool_stride=2
            ),
            ConvTimeBlock(
                in_channels=64, out_channels=128, pool_kernel_size=2, pool_stride=2
            ),
        )

    @staticmethod
    def lstm_block():
        """LSTM block."""
        return nn.Sequential(
            nn.Flatten(start_dim=2),  # keep batch, timestep dims
            GRUOutput(
                input_size=2048, hidden_size=64, bidirectional=True, batch_first=True
            ),
        )

    def linear_block(self):
        """Fully-connected block."""
        return nn.Sequential(
            nn.Dropout(self.drop),
            ReshapeOutput(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.drop),
            nn.Linear(64, 1),
        )

    def augment_pipe(self, choice, bs):
        """
        Data augmentation pipeline.

        Parameters
        ----------
        choice: int
            Random choice for augmentation.
        bs: int
            Batch size.

        Returns
        -------
        nn.Sequential
            Augmentation pipeline.
        """
        pipes = {
            0: nn.Sequential(
                nn.Identity(), MelSpectrogram(n_mels=self.n_mels, n_fft=self.n_fft)
            ),
            1: nn.Sequential(
                WhiteNoise(bs, self.snr_low, self.snr_high),
                MelSpectrogram(n_mels=self.n_mels, n_fft=self.n_fft),
            ),
            2: nn.Sequential(
                MelSpectrogram(n_mels=self.n_mels, n_fft=self.n_fft),
                FrequencyMasking(self.freq_mask),
            ),
            3: nn.Sequential(
                MelSpectrogram(n_mels=self.n_mels, n_fft=self.n_fft),
                TimeMasking(self.time_mask),
            ),
        }
        return pipes[choice]

    def get_results(self, spec, actual):
        """
        Get model results.

        Parameters
        ----------
        spec: torch.tensor
            Spectrogram.
        actual: float
            Actual emotion score.

        Returns
        -------
        Dict
            Dictionary of model results.
        """
        pred = self.model(spec)

        results = {
            "outputs": pred,
            "loss": self.loss_func(pred, actual),
            "MAE": mean_absolute_error(pred, actual),
            "MAPE": mean_absolute_percentage_error(pred, actual),
        }
        return results

    def training_step(self, batch, idx):
        """Training step."""
        x = batch[0].to(self.device)
        labels = batch[1]
        bs = x.shape[0]
        choice = random.randint(0, 3)
        make_spec = self.augment_pipe(choice, bs)
        spec = make_spec(x).unsqueeze(1)  # add channel dim to spec

        results = self.get_results(spec, labels)
        total_loss = results["loss"]

        self.log_dict(
            {
                "total_train_loss": total_loss,
                "train_loss": results["loss"],
                "train_mae": results["MAE"],
                "train_mape": results["MAPE"],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return total_loss

    def validation_step(self, batch, idx):
        """Validation step."""
        x = batch[0].to(self.device)
        labels = batch[1]
        make_spec = nn.Sequential(MelSpectrogram(n_mels=self.n_mels, n_fft=self.n_fft))
        spec = make_spec(x).unsqueeze(1)  # add channel dim to spec

        results = self.get_results(spec, labels)
        total_loss = results["loss"]

        self.log_dict(
            {
                "total_valid_loss": total_loss,
                "valid_loss": results["loss"],
                "valid_mae": results["MAE"],
                "valid_mape": results["MAPE"],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return total_loss

    def test_step(self, batch, idx):
        """Test step."""
        x = batch[0].to(self.device)
        labels = batch[1]
        raise NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Predict step."""
        x = batch[0].to(self.device)
        return self.forward(x)

    @staticmethod
    def preprocess(x):
        """Preprocess an audio tensor be applying VAD model."""
        if x.shape[-1] < 16000 * 8:
            raise ValueError(
                "Cannot preprocess tensor less than 8 seconds in duration."
            )
        vad = VadChunk(*get_vad("both"))
        return vad(x)

    def score_file(self, file, sample_rate=16000, vad=True):
        """
        Score a WAV file.

        Splits audio signal into separate 8-second chunks, individually scores each and returns
        mean across splits as final score.

        Parameters
        ----------
        file: Union[str, pathlib.Path]
            Filepath to audio file.
        sample_rate: int
            Sample rate of audio file; default 16,000hz
        vad: bool
            Whether to implement voice activity detection as preprocessing step; default True.

        Returns
        -------
        float
            Emotion score.
        """
        signal, _ = get_sample(file, sample_rate)
        return self.score_signal(signal, vad)

    def score_signal(self, x, vad=True):
        """
        Score a tensor.

        Splits audio signal into separate 8-second chunks, individually scores each and returns
        mean across splits as final score.

        Parameters
        ----------
        x: torch.Tensor
            Audio signal to score.
        vad: bool
            Whether to implement voice activity detection as preprocessing step; default True.

        Returns
        -------
        float
            Emotion score.

        """
        if vad:
            x = self.preprocess(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)  # need dim to match
        return self.forward(x)

    def forward(self, x):
        """
        Score a preprocessed tensor.

        Splits audio signal into separate 8-second chunks, individually scores each and returns
        mean across splits as final score.

        Parameters
        ----------
        x: torch.Tensor

        Returns
        -------
        float
            Emotion score.

        """
        splits = split_sample(x, n_seconds=8)
        spec = MelSpectrogram(n_mels=self.n_mels, n_fft=self.n_fft)
        samples = [spec(s).unsqueeze(1) for s in splits]
        self.model.eval()
        with torch.no_grad():
            preds = [self.model(s) for s in samples]
        return torch.tensor(preds).mean()

    def configure_optimizers(self):
        """Configure optimizer for training."""
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(opt),
                "monitor": "train_loss",
                "frequency": 1,
            },
        }
