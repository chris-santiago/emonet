"""
More miscellaneous functions that probably fit better somewhere else.
"""
import json
import pathlib
import random
from concurrent.futures import ThreadPoolExecutor
from functools import singledispatch
from statistics import mean
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
import torchaudio
from IPython.display import Audio, display
from speechbrain.dataio.encoder import CategoricalEncoder

from emonet import EMOTIONS, SAMPLE_RATE
from emonet.transforms import resample_rate

MAX_WORKERS = 50


def play_audio(waveform: torch.Tensor, sample_rate: int):
    """
    Play an audio signal within an IPython notebook.

    Parameters
    ----------
    waveform: torch.Tensor
        Audio signal.
    sample_rate: int
        Audio signal sample rate.

    Returns
    -------
    None
    """
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def get_metadata(file: pathlib.Path):  # todo finish once we have data
    with open(file, "r") as fp:
        return fp.readlines()


def async_file_operation(files: List, func: Callable, max_workers: int = MAX_WORKERS):
    """
    Asynchronously apply a function on a list of files.

    Parameters
    ----------
    files: List
        Files to apply function to.
    func: Callable
        Function to apply.
    max_workers: int
        Maximum number of concurrent threads.

    Returns
    -------
    None
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(func, files)


def get_rating_encoder(ratings: List[str]) -> CategoricalEncoder:
    """
    Get a categorical encoder for emotion severity labels.

    Parameters
    ----------
    ratings: List[str]
        List containing all possible categorical ratings.

    Returns
    -------
    CategoricalEncoder
        A fitted Speechbrain CategoricalEncoder object.
    """
    enc = CategoricalEncoder()
    enc.update_from_iterable(ratings)
    return enc


def binarize_ratings(
    ratings: Union[int, List, torch.Tensor]
) -> Union[int, torch.Tensor]:
    """
    Convert emotion severity levels to binary.

    Assigns `None` and `Low` (0 and 1, respectively) to `0`; assigns `Med` and `High`
    (2 and 3, respectively) to `1`.

    Parameters
    ----------
    ratings: Union[int, List, torch.Tensor]
        Rating(s) to convert; expects {0, 1, 2, 3}.

    Returns
    -------
    Union[int, torch.Tensor]
        Binarized label(s).
    """
    if isinstance(ratings, torch.Tensor):
        return (ratings > 1).long()
    return int(ratings > 1)


def binarize_labels(labels: Dict) -> Dict:
    """
    Convert all labels within a dictionary to binary.

    Parameters
    ----------
    labels: Dict
        A dictionary mapping a specific file or sample to its respective emotion severity label.

    Returns
    -------
    Dict
        The original `labels` dictionary with binarized labels.
    """
    return {k: binarize_ratings(v) for k, v in labels.items()}


@singledispatch
def weights_to_binary(weights):
    new = {}
    for emot in weights:
        new[emot] = [mean(weights[emot][i : i + 2]) for i in [0, 2]]
    return new


@weights_to_binary.register
def _(weights: dict):
    new = {}
    for emot in weights:
        new[emot] = [mean(weights[emot][i : i + 2]) for i in [0, 2]]
    return new


@weights_to_binary.register
def _(weights: list):
    return [mean(weights[i : i + 2]) for i in [0, 2]]


def get_sample(file: pathlib.Path, sample_rate: int) -> Tuple[torch.Tensor, int]:
    """
    Read an audio signal from a file and resample, if necessary.

    Parameters
    ----------
    file: pathlib.Path
        Path to audio file.
    sample_rate: int
        Desired sample rate.

    Returns
    -------
    Tuple[torch.Tensor, int]
        Audio signal (potentially resampled) and output sample rate.
    """
    waveform, in_rate = torchaudio.load(file)
    if in_rate != sample_rate:
        signal = resample_rate(waveform, in_rate, sample_rate)
        return signal.squeeze(), sample_rate  # downstream expecting 1-dim (speechbrain)
    return waveform.squeeze(), sample_rate  # downstream expecting 1-dim (speechbrain)


def get_random_segment(
    wav: torch.Tensor, seconds: int = 7, sample_rate: int = SAMPLE_RATE
) -> torch.Tensor:
    """
    Get a random n-second sample from an audio signal.
    Parameters
    ----------
    wav: torch.Tensor
        Original audio signal.
    seconds: int
        Desired duration of random sample.
    sample_rate: int
        Original audio sampling rate.

    Returns
    -------
    torch.Tensor
        A random segment of the original audio signal.
    """
    buffer = seconds * sample_rate
    end = (
        wav.shape[-1] - buffer
    )  # should pull timesteps if dims=1 or dims=2, provided following (batch, timestep, channel) format
    start = random.randint(0, end)
    if wav.ndim > 1:
        return wav[:, start : start + buffer]  # assumes (timestamp, channel)
    return wav[start : start + buffer]


def channel1to3(t: torch.Tensor) -> torch.Tensor:
    """
    Convert a single-channel spectrogram to 3-channels.

    Parameters
    ----------
    t: torch.Tensor
        3-d spectrogram (C, H, W)

    Returns
    -------
    torch.Tensor
        A 3-channel version of original spectrogram, where new channels are copies of original.
    """
    bs = t.shape[0]
    h, w = t.shape[-2:]
    if t.ndim < 4:
        t = t.unsqueeze(1)
    return t.expand((bs, 3, h, w))


def ohe_labels(labels, n_classes):
    try:
        y = torch.hstack(list(labels.values()))
    except TypeError:
        y = torch.tensor(list(labels.values()), dtype=torch.long)
    return F.one_hot(y, num_classes=n_classes)


def decode_ohe(labels, as_dict=False):
    labs = labels.argmax(-1).squeeze()
    if as_dict:
        return {emot: torch.Tensor([lab]).long() for emot, lab in zip(EMOTIONS, labs)}
    return labs


def from_json(filepath: pathlib.Path) -> Dict:
    """Read metadata from json file."""
    with open(filepath, "r") as fp:
        return json.load(fp)


def to_json(meta: Union[Dict, List], filepath: pathlib.Path):
    """Write metadata to json file."""
    with open(filepath, "w") as fp:
        json.dump(meta, fp, indent=2)
