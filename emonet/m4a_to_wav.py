"""
Convert m4a files to wav files.  Should only run once, initially.
"""
import pathlib
import shutil
import subprocess

from emonet import DATA_DIR
from emonet.log import make_logger
from emonet.utils import MAX_WORKERS, async_file_operation

WAVS = DATA_DIR.joinpath("wavs")
RAW_SAMPLES = DATA_DIR.joinpath("voice_labeling_report", "voice_samples")

# FFMPEG = "/opt/homebrew/bin/ffmpeg"  # todo probably don't need for most cases

logger = make_logger(__name__)


def m4a_to_wav(file: pathlib.Path):
    logger.info(f"Converting {file.name} to .wav ...")
    in_fn = str(file)
    out_fn = WAVS.joinpath(f"{file.stem}.wav")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", in_fn, out_fn]
    )


def copy_wav(file: pathlib.Path):
    dest = WAVS.joinpath(file.name)
    logger.info(f"Copying {file.name} to {dest.name} ...")
    shutil.copy(file, dest)


def main(max_workers=MAX_WORKERS):
    to_convert = list(pathlib.Path(RAW_SAMPLES).glob("*.m4a"))
    logger.info(f"Converting {len(to_convert)} files from .m4a to .wav ...")
    async_file_operation(to_convert, m4a_to_wav, max_workers)

    to_copy = list(pathlib.Path(RAW_SAMPLES).glob("*.wav"))
    logger.info(f"Copying {len(to_copy)} files to {WAVS} ...")
    async_file_operation(to_copy, copy_wav, max_workers)


if __name__ == "__main__":
    WAVS.mkdir(exist_ok=True)
    main()
