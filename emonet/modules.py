"""
Various NN modules.
"""
import logging
from typing import Callable, Optional, Tuple

import torch
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.processing.speech_augmentation import (
    AddNoise,
    DropChunk,
    DropFreq,
    SpeedPerturb,
)
from torch import nn as nn

from emonet import SAMPLE_RATE
from emonet.utils import get_random_segment


def get_vad(ret="model"):
    """Get pretrained Silero VAD model and/or utilities."""
    mod, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    obj = {"model": mod, "utils": utils, "both": (mod, utils)}
    try:
        return obj[ret]
    except KeyError:
        raise ValueError("Parameter `ret` must be one of {`model`, `utils`, `both`}")


class VadChunk(nn.Module):
    """Concatenate VAD chunks from signal."""

    def __init__(
        self, model: nn.Module, utils: Tuple[Callable], sample_rate: int = SAMPLE_RATE
    ):
        """
        Constructor method.

        Parameters
        ----------
        model: nn.Module
            Pre-trained VAD model.
        utils: Tuple[Callable]
            Tuple of utility functions.
        sample_rate: int
            Audio sample rate; default 16,000Hz
        """
        super().__init__()
        self.model = model
        self.get_speech_timestamps = utils[0]
        self.collect_chunks = utils[-1]
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through VAD model layer."""
        speech = self.get_speech_timestamps(
            x, self.model, sampling_rate=self.sample_rate
        )
        try:
            return self.collect_chunks(speech, x)
        except NotImplementedError:
            print("No voice activity detected.")
            return torch.zeros(1)


class RandomSegment(nn.Module):
    """Get a random segment from sample."""

    def __init__(self, seconds: int, sample_rate: int = SAMPLE_RATE):
        """
        Constructor method.

        Parameters
        ----------
        seconds: int
            Duration(seconds) of random segment to retrieve.
        sample_rate: int
            Audio sample rate; default 16,000Hz
        """
        self.seconds = seconds
        self.sample_rate = sample_rate
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through random segment model layer."""
        if x.shape[-1] <= self.seconds * self.sample_rate:
            return x  # can't sample if it's not big enough
        return get_random_segment(x, seconds=self.seconds, sample_rate=self.sample_rate)


class SBAugment(TimeDomainSpecAugment):
    """Do multiple Speech Brain augmentations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through random segment model layer."""
        x = x[
            :, :, None
        ]  # speechbrain expects tensor shape (batch, timesteps, channels)
        lengths = torch.ones(x.shape[0])
        x = self.speed_perturb(x)
        x = self.drop_freq(x)
        x = self.drop_chunk(x, lengths)
        return x.squeeze(-1)  # drop last dim


class TimeDistributed(nn.Module):
    """
    Apply a nn.Module across a time dimension.

    Expects a batch first tensor with shape (batch, time, channel, height, width), where time is the
    chunked spectrogram across time steps.

    The Pytorch-Forecasting version (https://pytorch-forecasting.readthedocs.io/en/stable/_modules/pytorch_forecasting/models/temporal_fusion_transformer/sub_modules.html)
    does not properly apply to images with shape (batch, channel, height, width) or (channel, height, width)/

    Source: https://github.com/Data-Science-kosta/Speech-Emotion-Classification-with-PyTorch/blob/master/notebooks/stacked_cnn_lstm.ipynb
    """

    def __init__(self, module: nn.Module):
        """
        Constructor method.

        Parameters
        ----------
        module: nn.Module
            Module to create time-distributed module/block from.
        """
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through time-distributed module layer."""
        if len(x.size()) <= 2:
            return self.module(x)
        # squash samples and timesteps into a single axis
        elif len(x.size()) == 3:  # (samples, timesteps, inp1)
            x_reshape = x.contiguous().view(
                -1, x.size(2)
            )  # (samples * timesteps, inp1)
        elif len(x.size()) == 4:  # (samples,timesteps,inp1,inp2)
            x_reshape = x.contiguous().view(
                -1, x.size(2), x.size(3)
            )  # (samples*timesteps,inp1,inp2)
        else:  # (samples,timesteps,inp1,inp2,inp3)
            x_reshape = x.contiguous().view(
                -1, x.size(2), x.size(3), x.size(4)
            )  # (samples*timesteps,inp1,inp2,inp3)

        y = self.module(x_reshape)

        # we have to reshape Y
        if len(x.size()) == 3:
            y = y.contiguous().view(
                x.size(0), -1, y.size(1)
            )  # (samples, timesteps, out1)
        elif len(x.size()) == 4:
            y = y.contiguous().view(
                x.size(0), -1, y.size(1), y.size(2)
            )  # (samples, timesteps, out1,out2)
        else:
            y = y.contiguous().view(
                x.size(0), -1, y.size(1), y.size(2), y.size(3)
            )  # (samples, timesteps, out1,out2, out3)
        return y


class ChunkDropper(nn.Module):
    """Drop time chunks from signal."""

    def __init__(self, bs: int, p: float = 1.0):
        """
        Constructor method.

        Parameters
        ----------
        bs: int
            Batch size used.
        p: float
            Probability of dropping chunk of signal.
        """
        super().__init__()
        self.bs = bs
        self.drop = DropChunk(drop_prob=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through chunk-dropping augmentation layer."""
        x = x.cuda()
        return self.drop(x, torch.ones(self.bs).cuda())


class FreqDropper(nn.Module):
    """Drop frequencies from signal."""

    def __init__(self, p: float = 1.0):
        """
        Constructor method.

        Parameters
        ----------
        p: float
            Probability of dropping chunk of signal.
        """
        super().__init__()
        self.drop = DropFreq(drop_prob=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through frequency-dropping augmentation layer."""
        x = x.cuda()
        return self.drop(x)


class Perturber(nn.Module):
    """Perturb signal speed."""

    def __init__(self, p: float = 1.0):
        """
        Constructor method.

        Parameters
        ----------
        p: float
            Probability of dropping frequency from signal.
        """
        super().__init__()
        self.perturb = SpeedPerturb(SAMPLE_RATE, perturb_prob=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through speed perturbation augmentation layer."""
        x = x.cuda()
        return self.perturb(x)


class WhiteNoise(nn.Module):
    """Add white noise to signal."""

    def __init__(self, bs: int, snr_low: int = 15, snr_high: int = 30):
        """
        Constructor method.

        Parameters
        ----------
        bs: int
            Batch size.
        snr_low: int
            Signal-to-noise ratio floor.
        snr_high: int
            Signal-to-noise ratio ceiling.
        """
        super().__init__()
        self.bs = bs
        self.noise = AddNoise(snr_low=snr_low, snr_high=snr_high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through white noise augmentation layer."""
        return self.noise(x, torch.ones(self.bs))


class LSTMOutput(nn.Module):
    """Grabs only output of LSTM module; discards hidden state stuff."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through LSTM layer."""
        output, _ = self.lstm(x)
        return output


class PrintShape(nn.Module):
    """Print layer shape for debugging."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get layer shape."""
        print(x.shape)
        return x


class LogMessage(nn.Module):
    """Log for debugging."""

    def __init__(self, logger: logging.Logger, msg: Optional[str] = None):
        """
        Constructor method.

        Parameters
        ----------
        logger: logging.Logger
            Program logger.
        msg: Optional[str]
            Optional message to send.
        """
        super().__init__()
        self.logger = logger
        self.msg = msg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Log message."""
        self.logger.info(f"{self.msg} - Shape: {x.shape}")
        return x


class ChunkSpectrogram(nn.Module):
    """Separate a spectrogram into n chunks."""

    def __init__(self, n_chunks: int = 6):
        """
        Constructor method.

        Parameters
        ----------
        n_chunks: int
            Number of chunks to split spectrogram into.
        """
        super().__init__()
        self.n_chunks = n_chunks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split spectrogram `x` ubti separate chunks,"""
        chunks = torch.chunk(
            x, self.n_chunks, -1
        )  # in dims (batch, channel, height, width)
        return torch.stack(chunks, 1)  # out dims (batch, time, channel, height, width)


class GRUOutput(nn.Module):
    """Grab only output of GRU module; discard hidden state stuff."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gru = nn.GRU(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through GRU layer."""
        output, _ = self.gru(x)
        return output


class ReshapeOutput(nn.Module):
    """Reshape LSTM output to remove timestep dimension."""

    def __init__(self, method: str = "mean"):
        super().__init__()
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through reshaping layer."""
        if self.method == "last":
            return x[:, -1, :]
        if self.method == "mean":
            return x.mean(1)
        raise ValueError("Method must be one of {`mean`, `last`}")
