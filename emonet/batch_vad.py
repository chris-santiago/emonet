"""
Module to apply pre-train VAD model to files.

Intended to run as batch process before any training begins.
"""
import pathlib

import torch
import torch.nn as nn
import torchaudio
import tqdm

from emonet import DATA_DIR, SAMPLE_RATE
from emonet.log import make_logger
from emonet.utils import get_sample

logger = make_logger(__name__)

WAVS = DATA_DIR.joinpath("wavs")
VAD_WAVS = DATA_DIR.joinpath("vad_wavs")
VAD_WAVS.mkdir(exist_ok=True)

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)

get_speech_timestamps, collect_chunks = vad_utils[0], vad_utils[-1]


class VadChunk:
    """
    Class for applying pre-trained VAD model to audio file.
    """

    def __init__(self, model: nn.Module, sample_rate: int = SAMPLE_RATE):
        """
        Constructor method.

        Parameters
        ----------
        model: nn.Module
            Pre-trained voiced activity detection (VAD) model.
        sample_rate: int
            Sample rate of audio file.
        """
        self.model = model
        self.sample_rate = sample_rate

    def get(self, x: torch.Tensor, filename: pathlib.Path) -> torch.Tensor:
        """
        Get speech chunks consolidated into single tensor.

        Parameters
        ----------
        x: torch.Tensor
            Original audio signal.
        filename: pathlib.Path
            Filename containing original audio for tensor `x`.
        Returns
        -------
        torch.Tensor
            VAD chunks consolidated into single tensor.
        """
        try:
            speech = get_speech_timestamps(
                x, self.model, sampling_rate=self.sample_rate
            )
            return collect_chunks(speech, x)
        except Exception as e:
            print(e)
            print(f"Check {filename}. VAD did not output data.")


def main():
    """Run VAD for all wav files in data directory."""
    files = list(WAVS.glob("*.wav"))
    filenames = [x.name for x in files]
    logger.info(f"Applying VAD model to {len(filenames)} files ...")
    vad = VadChunk(vad_model)
    samples = map(lambda x: get_sample(x, SAMPLE_RATE)[0], files)  # type: ignore
    speech = map(vad.get, samples, filenames)

    for i, wav in tqdm.tqdm(enumerate(speech), total=len(files)):
        logger.info(f"Trying {filenames[i]}")
        try:
            torchaudio.save(
                VAD_WAVS.joinpath(files[i].name), wav.unsqueeze(0), SAMPLE_RATE
            )
        except Exception as e:
            logger.warning(e)
            logger.warning(f"Check {files[i].name}. VAD did not output data.")


if __name__ == "__main__":
    main()
