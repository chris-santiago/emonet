"""
This module is not for unit testing. Here we're testing that there's no overlap within the
train/valid/test data manifests.
"""
import pytest

from emonet import DATA_DIR, THERAPISTS
from emonet.utils import from_json

DATASETS = ['train', 'valid', 'test']


def test_full_wavs_unique():
    file_names = {}
    for ds in DATASETS:
        manifest = from_json(DATA_DIR.joinpath(f'{ds}.json'))
        file_names[ds] = set([x['file_name'] for x in manifest.values()])
    assert len(file_names['train'].intersection(file_names['valid'])) == 0
    assert len(file_names['train'].intersection(file_names['test'])) == 0
    assert len(file_names['valid'].intersection(file_names['test'])) == 0


def test_vad_splits_unique():
    file_names = {}
    for ds in DATASETS:
        manifest = from_json(DATA_DIR.joinpath(f'{ds}_splits.json'))
        file_names[ds] = set([x['file_name'] for x in manifest.values()])
    assert len(file_names['train'].intersection(file_names['valid'])) == 0
    assert len(file_names['train'].intersection(file_names['test'])) == 0
    assert len(file_names['valid'].intersection(file_names['test'])) == 0


@pytest.mark.parametrize(
    'therapist',
    THERAPISTS,
    ids=THERAPISTS
)
def test_therapist_splits_unique(therapist):
    file_names = {}
    for ds in DATASETS:
        manifest = from_json(DATA_DIR.joinpath(therapist, f'{ds}_splits.json'))
        file_names[ds] = [x['file_name'] for x in manifest.values()]
    assert file_names['train'] not in file_names['valid']
    assert file_names['train'] not in file_names['test']
    assert file_names['valid'] not in file_names['test']
