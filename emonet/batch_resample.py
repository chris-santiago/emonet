"""Module for resampling audio files in batch operation."""
import pathlib

import torchaudio

from emonet import DATA_DIR, SAMPLE_RATE
from emonet.log import make_logger
from emonet.transforms import resample_rate
from emonet.utils import MAX_WORKERS, async_file_operation

RESAMPLE_RATE = 16000

logger = make_logger(__name__)


def resample(file: pathlib.Path) -> None:
    """
    Resample an audio file to 16,000hz.

    Parameters
    ----------
    file: pathlib.Path
        Path to audiofile.

    Returns
    -------
    None
        Opens original file, resamples and overwrites original audio.
    """
    orig, in_rate = torchaudio.load(file)
    if in_rate != RESAMPLE_RATE:
        logger.info(f"Resampling {file.name} to {RESAMPLE_RATE}")
        resampled = resample_rate(orig, in_rate, SAMPLE_RATE)
        torchaudio.save(file, resampled, RESAMPLE_RATE)
    pass


def main(max_workers: int = MAX_WORKERS) -> None:
    """
    Resample all files in the data directory.

    Parameters
    ----------
    max_workers: int
        Maximum number of asynchronous processes.

    Returns
    -------
    None
        Resamples and overwrites original audio file.
    """
    wavs = list(DATA_DIR.joinpath("wavs").glob("*.wav"))
    async_file_operation(wavs, resample, max_workers)


if __name__ == "__main__":
    main()
