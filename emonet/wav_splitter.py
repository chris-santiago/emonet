"""
Module for splitting existing wavs into smaller files.
Assumes `m4ato_wav.py` and `data_prep.py` have already been run, as it requires existing wav files
and training manifests.
"""
from typing import List

import torch
import torchaudio
import tqdm

from emonet import DATA_DIR, SAMPLE_RATE
from emonet.log import make_logger
from emonet.utils import from_json, get_sample, to_json

logger = make_logger(__name__)

THERAPISTS = ["Michelle Lyn", "Pegah Moghaddam", "Sedara Burson", "Yared Alemu"]
SPLITS = DATA_DIR.joinpath("vad_splits")  # !! pay attention to me!
DURATION = 8


def split_sample(
    signal: torch.Tensor, n_seconds: int, sample_rate: int = SAMPLE_RATE, dim: int = -1
) -> List:
    """
    Split an audio signal into multiple n-second chunks.

    Discards chunks with duration < `n_seconds`.

    Parameters
    ----------
    signal: torch.Tensor
        Audio signal.
    n_seconds: int
        Desired duration of sample chunks.
    sample_rate: int
        Audio signal sample rate; default 16000
    dim: int
        Dimension to split sample on; defaults to last.

    Returns
    -------
    List[torch.Tensor]
        List of equal-length audio samples.
    """
    split_length = n_seconds * sample_rate
    splits = torch.split(signal, split_length, dim=dim)
    return [s for s in splits if s.size()[-1] == split_length]


def split_files(ds: str = "train", therapist: str = None) -> None:
    """
    Split multiple samples in large batch.

    Iterates through therapist-specific dataset manifest to read in respective audio files
    as tensors, apply splits, and write splits to `.wav` files. Creates new `split` manifests,
    complete with original metadata.

    Parameters
    ----------
    ds: str
        Dataset to run batch operation; should be one of {`train`, `valid`, `split`}.
    therapist: str
        Optional therapist flag.
    Returns
    -------
    None
        Output written to respective `.wav` files; manifest written to respective `.json` files.
    """
    if therapist:
        logger.info(f"Now splitting {ds} wavs for: {therapist}...")
        meta_file = DATA_DIR.joinpath(therapist, f"{ds}.json")
    else:
        logger.info(f"Now splitting {ds} wavs ...")
        meta_file = DATA_DIR.joinpath(f"{ds}.json")

    meta = from_json(meta_file)
    manifest = {}
    for wav_key in tqdm.tqdm(meta):
        wav_file = DATA_DIR.joinpath(meta[wav_key]["file_path"])
        signal, _ = get_sample(wav_file, SAMPLE_RATE)
        logger.info(f"Splitting {wav_file.name} into {DURATION} second chunks ...")
        splits = split_sample(signal, n_seconds=DURATION)
        for i, split in enumerate(splits):
            split_name = f"{wav_file.stem}--{i}"
            file_name = f"{split_name}.wav"
            file_path = SPLITS.joinpath(file_name)
            manifest[split_name] = meta[wav_key].copy()
            manifest[split_name]["file_path"] = str(file_path.relative_to(DATA_DIR))
            manifest[split_name]["file_name"] = file_path.name
            manifest[split_name]["stem"] = file_path.stem
            manifest[split_name]["duration"] = DURATION
            torchaudio.save(file_path, split.unsqueeze(0), SAMPLE_RATE)
    if therapist:
        json_out = DATA_DIR.joinpath(therapist, f"{ds}_splits.json")
    else:
        json_out = DATA_DIR.joinpath(f"{ds}_splits.json")
    to_json(manifest, json_out)


def split_files_therapist(ds: str = "train") -> None:
    """
    Split multiple samples in large batch.

    Iterates through therapist-specific dataset manifest to read in respective audio files
    as tensors, apply splits, and write splits to `.wav` files. Creates new `split` manifests,
    complete with original metadata.

    Parameters
    ----------
    ds: str
        Dataset to run batch operation; should be one of {`train`, `valid`, `split`}.

    Returns
    -------
    None
        Output written to respective `.wav` files; manifest written to respective `.json` files.
    """
    for therapist in THERAPISTS:
        split_files(ds, therapist)


def main():
    """Splite all files for train/valid/test datasets."""
    for ds in ["train", "valid", "test"]:  # todo may want to leave test out
        split_files(ds)


if __name__ == "__main__":
    SPLITS.mkdir(exist_ok=True)
    main()
