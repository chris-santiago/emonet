"""
Run once, initially, to setup expected data directory within your home folder.
"""

import pathlib
import shutil

from emonet import HERE
from emonet.log import make_logger

logger = make_logger(__name__)

HOME = pathlib.Path("~").expanduser()
DATA_DIR = HOME.joinpath("emonet-data")


def initialize():
    """Initialize data directory."""
    dirs = [("emonet-data", "wavs"), ("emonet-data", "voice_labeling_report")]
    for d in dirs:
        new_dir = HOME.joinpath(*d)
        pathlib.Path.mkdir(new_dir, exist_ok=True, parents=True)


def copy_manifests():
    """Copy manifests to data directory."""
    manifest_fp = HERE.joinpath("manifests")
    manifest_files = [*manifest_fp.glob("*")]
    for file in manifest_files:
        dest_path = DATA_DIR.joinpath(file.name)
        shutil.copy(file, dest_path)


def main():
    """Initialized data directory and copy relevant data manifest(s)."""
    logger.info(f"Initializing data directory in {DATA_DIR} ...")
    initialize()


if __name__ == "__main__":
    main()
    # copy_manifests()
