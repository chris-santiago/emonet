"""
Module for creating dataset with augmentation.  Good working version.
"""
import pathlib
from typing import Callable

import torch
from torch.utils.data import Dataset

from emonet import EMOTIONS, RATINGS
from emonet.utils import from_json, get_rating_encoder, get_sample


class TQDataset(Dataset):  # todo add sorting
    """
    Base Dataset class for emonet data.
    """

    def __init__(
        self,
        meta_data: pathlib.Path,
        data_dir: pathlib.Path,
        transform: Callable = None,
        target_transform: Callable = None,
        min_duration: int = None,
        emotion: str = None,
    ):
        """
        Construct a dataset.

        Parameters
        ----------
        meta_data: pathlib.Path
            Path to metadata.
        data_dir: pathlib.Path
            Path to data directory.
        transform: Callable
            Transformer function for input data.
        target_transform: Callable
            Transformer function for label data.
        min_duration: int
            Minimum duration (seconds) to filter audio files.
        emotion: str
            Emotion to retrieve labels/score for.
        """
        if min_duration:
            self.meta = {
                k: v
                for k, v in from_json(meta_data).items()
                if v["duration"] > min_duration
            }
        else:
            self.meta = from_json(meta_data)
        self.keys = list(self.meta.keys())
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.emotion = emotion
        self.min_duration = min_duration
        self.ratings = get_rating_encoder(RATINGS)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.meta[key]
        file = self.data_dir.joinpath(item["file_path"])
        signal, _ = get_sample(file=file, sample_rate=item["sample_rate"])
        if self.transform:
            signal = self.transform(signal)
        labels = {emot: self.ratings.lab2ind[item[emot]] for emot in EMOTIONS}
        if self.target_transform:
            labels = self.target_transform(labels)
        # return {'signal': signal, 'labels': labels}  # todo maybe drop to tuple from dict
        if self.emotion:
            return signal, labels[self.emotion]
        return signal, labels


class TQRegressionDataset(TQDataset):
    """
    Dataset class for running emonet data as regression. Outputs an average score as training
    input vice an emotion(s) and label(s).
    """

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.meta[key]
        file = self.data_dir.joinpath(item["file_path"])
        signal, _ = get_sample(file=file, sample_rate=item["sample_rate"])
        if self.transform:
            signal = self.transform(signal)
        score = item["avg_score"]
        if self.target_transform:
            score = self.target_transform(score)
        if self.emotion:
            return signal, torch.tensor([score[self.emotion]])
        return signal, score


class TQSplitDataset(TQDataset):
    """
    Idea here was to split a single audio sample into multiple n-second samples.
    """

    def split_sample(self, sample, labels, length):
        splits = [
            (t, labels) for t in torch.split(sample, length) if t.size(-1) == length
        ]
        if len(splits) > 24:  # effectively limit batch size
            splits = splits[:24]
        tensors = torch.stack([x[0] for x in splits])
        label_list = [x[1] for x in splits]
        labels_out = {k: torch.Tensor() for k in labels}
        for label_set in label_list:
            for emot in label_set:
                labels_out[emot] = torch.hstack(
                    [labels_out[emot], torch.Tensor([label_set[emot]])]
                ).long()
        return tensors, labels_out

    def __getitem__(self, idx):
        key = self.keys[idx]
        item = self.meta[key]
        file = self.data_dir.joinpath(item["file_path"])
        signal, sample_rate = get_sample(file=file, sample_rate=item["sample_rate"])
        labels = {emot: self.ratings.lab2ind[item[emot]] for emot in EMOTIONS}
        if self.transform:
            signal = self.transform(signal)
        split_length = self.min_duration * sample_rate
        return self.split_sample(signal, labels, split_length)


if __name__ == "__main__":
    # !! testing/debugging
    from torch.utils.data import DataLoader

    from emonet import DATA_DIR
    from emonet.utils import binarize_labels

    # tsfmr = nn.Sequential(
    #     RandomSegment(seconds=5),
    #     SBAugment(perturb_prob=0.2, drop_freq_prob=0.2, drop_chunk_prob=0.2, speeds=[100])
    # )
    # ds = TQSplitDataset(DATA_DIR.joinpath('Michelle Lyn', 'train.json'), DATA_DIR, min_duration=5)
    # dl = DataLoader(ds, None)
    # train = TQDataset(DATA_DIR.joinpath('Michelle Lyn', 'train_splits.json'), DATA_DIR,
    #                   target_transform=binarize_labels, emotion='fear')
    # data = get_datasets('Michelle Lyn', DATA_DIR, min_duration=5, split=True)
    train = TQRegressionDataset(
        DATA_DIR.joinpath("train_splits.json"), DATA_DIR, emotion="anger"
    )
    dl = DataLoader(train, 2)
    it = iter(dl)
    batch = next(it)
    print(batch)
    print(batch[0].dtype)
    print(batch[1].dtype)
