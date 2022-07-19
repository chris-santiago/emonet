"""
Spectrogram function examples from torchaudio.
"""
import torch
from torchaudio import transforms as T

from emonet import SAMPLE_RATE

RESAMPLE_RATE = 16000


def resample_rate(
    waveform: torch.Tensor, orig_freq: int, new_freq: int = RESAMPLE_RATE, **kwargs
) -> torch.Tensor:
    """

    Parameters
    ----------
    waveform: torch.Tensor
        Audio signal to resample.
    orig_freq: int
        Original sample frequency.
    new_freq: int
        Desired sample frequency.

    Returns
    -------
    torch.Tensor
        Resampled audio signal.
    """
    resample = T.Resample(orig_freq=orig_freq, new_freq=new_freq, **kwargs)
    return resample(waveform)


def get_spectrogram(
    waveform,
    n_fft=400,  # corresponds with 25ms window on 16,000hz sample
    win_len=None,
    hop_len=None,
    power=2.0,
):
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)


def get_melspectrogram(
    waveform,
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    win_len=None,
    hop_len=None,
    power=2.0,
):
    spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)
