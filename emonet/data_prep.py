"""
Preparing data manifests dataset creation.

This only needs to be run once, initially.  Assumes that you have metadata and wav files within
a `emonet-data` directory in your home folder; wavs should be in `~/emonet-data/wavs` or in
`~/emonet-data/vad_wavs` if you've run VAD model..

Intent is to create a master dataset manifest, including paths to files, emotion ratings by therapist,
sample duration, and other metadata. Master manifest is further broken into train/valid/test manifests
for each therapist.

Sets seed for reproducibility.
"""
import pathlib
import random
from typing import Dict, List, Optional, Tuple, Union

from emonet import DATA_DIR, RATINGS, SAMPLE_RATE
from emonet.log import make_logger
from emonet.utils import from_json, get_rating_encoder, get_sample, to_json

logger = make_logger(__name__)

SEED = 42
logger.info(f"Using seed {SEED}.")
DEFAULT_FILE = DATA_DIR.joinpath("voice_labeling_report", "voice_labels.json")
WAVS = DATA_DIR.joinpath(
    "vad_wavs"
)  # todo it might make sense to iterate through `wavs` and `vad_wavs`
THERAPISTS = ["Michelle Lyn", "Pegah Moghaddam", "Sedara Burson", "Yared Alemu"]
SPLIT_RATIO = (80, 10, 10)
MULTI_SPEAKERS = ["SK", "RDR", "YH", "GS", "SRC"]


def get_metadata(
    files: List[pathlib.Path],
    labels_fn: pathlib.Path = DEFAULT_FILE,
    data_dir: pathlib.Path = DATA_DIR,
) -> Dict:
    """
    Get metadata from audio files, compile into manifest.

    Parameters
    ----------
    files: List[pathlib.Path]
        List of files to include in manifest.
    labels_fn: pathlib.Path
        File containing original voice labels/report.
    data_dir: pathlib.Path
        Directory containing all data files.
    Returns
    -------
    Dict
        Dictionary mapping a unique file ID (stem) to respective metadata and labels.
    """
    logger.info(f"Grabbing metadata from {len(files)} files ...")
    meta = {}
    for file in files:
        meta.update(get_entry(file, data_dir))
    labels = get_labels_by_id(labels_fn)
    for id, data in meta.items():
        data.update(labels[id])
    return meta


def get_entry(file: pathlib.Path, data_dir=DATA_DIR) -> Dict:
    """
    Create a dictionary entry based on file metadata.
    Parameters
    ----------
    file: pathlib.Path
        Path to audio file.
    data_dir: pathlib.Path
        Directory containing all data files.

    Returns
    -------
    Dict
        Dictionary mapping file stem to file metadata.
    """
    logger.info(f"Creating manifest entry for {file.name} ...")
    signal, sample_rate = get_sample(file, SAMPLE_RATE)
    return {
        file.stem: {
            "file_path": str(file.relative_to(data_dir)),
            "file_name": file.name,
            "sample_rate": sample_rate,
            "duration": signal.shape[0] / sample_rate,
            "stem": file.stem,
        },
    }


def get_labels_by_id(filepath: pathlib.Path, therapist: Optional[str] = None) -> Dict:
    """
    Get emotion labels for each file within a metadata manifest.

    Parameters
    ----------
    filepath: pathlib.Path
        Path to metadata file.
    therapist: Optional[str]
        Specific therapist to retrieve labels from. Default `None` return all therapist labels.
    Returns
    -------
    Dict
        A dictionary of file keys mapped to labels.
    """
    meta = from_json(filepath)
    if therapist:
        check_therapist(therapist)
        logger.info(f"Getting emotion labels for {therapist} ...")
        return {k.split(".")[0]: v.get(therapist) for k, v in meta.items()}
    logger.info("Getting emotion labels for manifest ...")
    return {k.split(".")[0]: v for k, v in meta.items()}


def prepare_data(
    metadata: Union[pathlib.Path, Dict],
    therapist: Optional[str] = None,
    split_ratio: Union[List, Tuple] = SPLIT_RATIO,
    data_dir: pathlib.Path = DATA_DIR,
    return_dict: bool = False,
) -> Optional[Dict]:
    """
    Prepare data for training/validation/testing.

    Parameters
    ----------
    metadata: Union[pathlib.Path, Dict]
        Dataset metadata.
    therapist: Optional[str]
        Optional therapist to filter on.
    split_ratio: Union[List, Tuple]
        Train/valid/test split ratio.
    data_dir: pathlib.Path
        Path to data directory.
    return_dict: bool
        Whether to return dictionary or not; if not returned output exported to .json file.

    Returns
    -------
    Optional[Dict]
        Dictionary containing data manifest or None, depending on call args.
    """
    if isinstance(metadata, pathlib.Path):
        meta = from_json(metadata)
    elif isinstance(metadata, Dict):
        meta = metadata
    else:
        raise ValueError("Expected either `pathlib.Path` or `dict` object.")
    meta = filter_bad_files(meta)  # remove bad files before splitting
    data = split_sets(meta, split_ratio, data_dir)
    if therapist:
        data_dir.joinpath(therapist).mkdir(parents=True, exist_ok=True)
        file_names = {k: data_dir.joinpath(therapist, f"{k}.json") for k in data.keys()}
    else:
        file_names = {k: data_dir.joinpath(f"{k}.json") for k in data.keys()}
    for key, files in data.items():
        filtered = filter_meta(files, meta)
        if return_dict:
            return filtered
        else:
            to_json(filtered, file_names[key])


def filter_bad_files(meta: Dict) -> Dict:
    """
    Drop bad files from manifest.

    Parameters
    ----------
    meta: Dict
        Dataset metadata.

    Returns
    -------
    Dict
        Dictionary of metadata with bad files removed.
    """
    keys = list(meta.keys())
    bad = [k for k in keys if any(key in k for key in MULTI_SPEAKERS)]
    logger.info(f"Dropping {len(bad)} files with multiple speakers ...")
    return {k: v for k, v in meta.items() if k not in bad}


def filter_meta(files: List, meta: Dict) -> Dict:
    """
    Remove files from metadata manifest not included in dataset.

    Parameters
    ----------
    files: List
        List of files included in dataset.
    meta: Dict
        Dictionary of dataset metadata.

    Returns
    -------
    Dict
        Dictionary containing metadata of only audio in `files` arg.
    """
    files = [f.name for f in files]
    return {k: v for k, v in meta.items() if v["file_name"] in files}


def split_sets(
    meta: Dict,
    split_ratio: Union[List, Tuple] = SPLIT_RATIO,
    data_dir: pathlib.Path = DATA_DIR,
) -> Dict:
    """
    Split data manifest into train/valid/test splits.

    Parameters
    ----------
    meta: Dict
        Master manifest.
    split_ratio: Tuple[int]
        Tuple containing split percents; default (80, 10, 10).
    data_dir: pathlib.Path
        Directory containing all data files.

    Returns
    -------
    Dict
        Dictionary containing train/valid/test dataset metadata.
    """
    files = [data_dir.joinpath(v["file_name"]) for v in meta.values()]
    logger.info("Shuffling files for train/valid/test splits ...")
    random.seed(SEED)
    random.shuffle(files)
    logger.info(f"Splitting files with {SPLIT_RATIO} ratio ...")
    tot_split = sum(split_ratio)
    tot_snts = len(files)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = files[0:n_snts]
        del files[0:n_snts]
    data_split["test"] = files
    return data_split


def check_therapist(therapist: str) -> None:
    """Check valid therapist name passed."""
    if therapist not in THERAPISTS:
        raise ValueError(f"Therapist must be in {THERAPISTS}")


def check_labels(labels: Dict) -> None:  # todo not currently used; but useful
    """Check files for number of labels/ratings."""
    for id, ratings in labels.items():
        if len(ratings) < 4:
            logger.warning(f"{id} has only {len(ratings)} scores.")
        for therapist, emotions in ratings.items():
            if len(emotions) < 5:
                logger.warning(
                    f"{id} - {therapist} labled only {len(emotions)} emotions."
                )


def get_therapist_metadata(meta: Dict, therapist: str) -> Dict:
    """
    Get therapist-specific metadata.

    Parameters
    ----------
    meta: Dict
        Dataset metadata.
    therapist: str
        Therapist to filter data on.

    Returns
    -------
    Dict
        Dictionary of therapist-specific metadata.
    """
    logger.info(f"Grabbing metadata for {therapist} ...")
    filtered = {k: v for k, v in meta.items() if therapist in v.keys()}
    new = {}
    for key, val in filtered.items():
        new[key] = {
            "file_path": val["file_path"],
            "file_name": val["file_name"],
            "sample_rate": val["sample_rate"],
            "duration": val["duration"],
            "stem": val["stem"],
            "therapist": therapist,
        }
        new[key].update(val[therapist].items())
        # for k, v in val[therapist].items():
        #     new[k] = v
    return new


def to_records(
    meta: Dict, therapists: List[str] = THERAPISTS
) -> List[Dict]:  # todo not used, but useful
    """
    Reformat metdata into a list of records.

    Parameters
    ----------
    meta: Dict
        Dataset metadata.
    therapists: List[str]
        List of therapists.
    Returns
    -------
    List[Dict]
        List of metadata records.
    """
    records = []
    for therapist in therapists:
        for key, data in meta.items():
            entry = {"key": key, "duration": data["duration"], "therapist": therapist}
            entry.update(data.get(therapist, {}))
            records.append(entry)
    return records


def filter_splits(therapist: str) -> None:
    """Filter existing train/valid/test manifests within therapist folders."""
    for split in ["train", "valid", "test"]:
        file = DATA_DIR.joinpath(therapist, f"{split}_splits.json")
        meta = from_json(file)
        filtered = filter_bad_files(meta)
        to_json(filtered, file)


def get_avg_score(item: Dict, emotion: str) -> float:
    """
    Get average emotional severity across therapists.

    Used for regression task.

    Parameters
    ----------
    item: Dict
    """
    enc = get_rating_encoder(RATINGS)
    ratings = [item.get(t) for t in THERAPISTS if item.get(t)]  # Don't want None values
    # num_scores = len(ratings)  # todo had though about normalizing across ALL possible therapists, vice ones that scored
    scores = [
        enc.lab2ind[val] for d in ratings for key, val in d.items() if key == emotion
    ]
    return sum(scores) / len(scores)


def get_metadata_from_files():
    """Iterate through all wav files to create metadata manifest."""
    files = sorted(list(WAVS.glob("*.wav")))
    out_fn = DATA_DIR.joinpath("metadata.json")
    meta = get_metadata(files)
    to_json(meta, out_fn)


def make_therapist_train_valid_test_sets(remove_splits: bool = False):
    """Make train/valid/test sets per therapist."""
    meta = from_json(DATA_DIR.joinpath("metadata.json"))
    for therapist in THERAPISTS:
        filtered = get_therapist_metadata(meta, therapist)
        prepare_data(filtered, therapist)

    if remove_splits:
        # removing bad files from wav-splits manifests
        # don't need to do if recreating from scratch; this was an after-fact fix
        for therapist in THERAPISTS:
            filter_splits(therapist)


def add_avg_scores_to_meta():
    """Add average scores data to metadata manifest."""
    meta = from_json(DATA_DIR.joinpath("metadata.json"))
    out_fn = DATA_DIR.joinpath("metadata.json")

    # adding average score
    new = {k: v for k, v in meta.items()}
    for key, item in meta.items():
        entry = {
            "avg_score": {
                "anger": get_avg_score(item, "anger"),
                "fear": get_avg_score(item, "fear"),
                "sadness": get_avg_score(item, "sadness"),
            }
        }
        new[key].update(entry)
    to_json(new, out_fn)


def make_main_train_valid_test_sets() -> None:
    """Make train/valid/test sets for full wav files."""
    logger.info("Making train/valid/test datasets for full wav files.")
    meta = from_json(DATA_DIR.joinpath("metadata.json"))
    prepare_data(meta)


def make_therapist_splits_train_valid_test_sets():
    """Use per therapist splits to make aggregate splits."""
    logger.info("Making train/valid/test datasets for split wav files.")
    master = from_json(DATA_DIR.joinpath("metadata.json"))
    for split in ["train", "valid", "test"]:
        agg = {}
        for therapist in THERAPISTS:
            file = DATA_DIR.joinpath(therapist, f"{split}_splits.json")
            meta = from_json(file)
            for key, val in meta.items():
                root = key.split("--")[0]
                agg[key] = {
                    "file_path": val["file_path"],
                    "file_name": val["file_name"],
                    "sample_rate": val["sample_rate"],
                    "duration": val["duration"],
                    "stem": val["stem"],
                    "avg_score": master[root].get("avg_score"),
                }
        fp = DATA_DIR.joinpath(f"{split}_splits.json")
        to_json(agg, fp)


def main():
    """Run data preparation."""
    # grabbing all metadata
    get_metadata_from_files()
    add_avg_scores_to_meta()
    make_main_train_valid_test_sets()


if __name__ == "__main__":
    main()
