# emonet

[![DOI](https://zenodo.org/badge/515677040.svg)](https://zenodo.org/badge/latestdoi/515677040)

A package to model negative emotion severity in patients with adverse childhood events (ACE).

**Contributors**: [@chris-santiago](https://github.com/chris-santiago), [@gonzalezeric](https://github.com/gonzalezeric), [@costargc](https://github.com/costargc)

## Installation & Environment Setup

### Using Docker

1. Open Terminal on Mac or PowerShell on Windows from the **root directory**.
2. When the application is open, in the command line, build the image using the following: `docker build -t cjsantiago/emonet-model -f docker/model/Dockerfile .`
3. Run container `docker run -it --name emonet -p 8888:8888 -v "${HOME}"/emonet-data:/home/jovyan/emonet-data  cjsantiago/emonet-model`
   - **Important**: Training the model or running batch inference presumes that you have a `emonet-data` directory within your home folder, containing the original `voice_labeling_report` directory. This will allow you to replicate all batch preprocessing done prior to model training.
   - You can score file(s) or signal(s), either on their own or with your own custom DataLoader, without the data directory (described above).
   - See `docker/model/README.md` for more.
4. Once the container has been created, you may access the files using one of the URLs generated in the CLI. 

### Using Conda

1. Clone this repo, then `cd emonet`
2. Create a virtual environment
   - For training and notebooks, use `conda env create -f env-base-plus.yml`
   - For scoring, only, use `conda env create -f env-base-yml`
3. Install `emonet` package, `pip install -e .`

**NOTE**: We're installing in editable mode (`-e` flag) as we expect to run training and/or scoring
from this cloned repo. Editable mode will symlink source code from the cloned repo directory to the
appropriate Python interpreter, enabling source code edits and easy-access to our saved models under
the `saved_models` directory.

#### Installing ffmpeg

`ffmpeg` is required to convert `.m4a` to `.wav`. On Mac this can be installed via [Homebrew](https://formulae.brew.sh/formula/ffmpeg).  *Skip this if you're running via Docker.*

## Data Setup

To use our original datset splits, we recommend downloading directly from our S3 bucket. This also
removes the need to complete some time-consuming preprocessing steps.

### Download from S3

*Assumes that you have the AWS CLI tool installed on your machine (and that you have our credentials :grinning:)*.

Within your home folder, create a directory called `emonet-data`. You could also use our directory
setup script `python emonet/dir_setup.py`.

From the `emonet-data` directory, run this command to sync (copy) the required directories and files
directly from our S3 bucket.

*Note that this assumes our credentials are located within `~/.aws/credentials`*

```bash
aws s3 sync s3://gatech-emonet/eval/ .
```

### From Scratch

Once you've setup the environment and installed the `emonet` package:

Run `python emonet/data_setup.py`

**Note:** you can pass an optional number of max_workers to this command; the default is 8 (threads).

`python emonet/data_setup.py 16`

This script will run and perform the following:

1. dir_setup.py: Set up a data directory within the home folder
2. m4a_to_wav.py: Convert any `.m4a` files to `.wav`
3. batch_resample.py: Resample files to 16,000Hz
4. batch_vad.py: Run voice activity detection (VAD)
5. data_prep.py: Create train/valid/test splits and respective manifests
6. wav_splitter.py: Split `.wav` files into 8-second chunks, the create new train/valid/test manifests that use the chunked `.wav` files

## Training

Now that files have all been converted to WAV, preprocessed with VAD and split training, validation and 
testing sets, and chunked into 8-second segments:

### Command Line Tool
The easiest way to run the model is via the CLI:

Run 

```bash
python emonet/train.py <num_workers> <max_epochs> <emotion>
```

and pass in the desired number of `workers`, `epochs` and `emotion`.

Example:

```bash
python train.py 12 300 anger
```

### Python
You can also train the model in Python:

Open a .py file or notebook and run

```python
from emonet import DATA_DIR
from emonet.train import main

main(workers=12, epochs=300, emotion="sadness", use_wandb=False, data_dir=DATA_DIR)
```

and pass in the desired number of `workers`, `epochs` and `emotion`; you can log runs to Weights &
Biases by setting `use_wandb=True`, and change the default data directory using the `data_dir` parameter.

## Pretrained Models

**No pretrained models are included in this public-facing repo.**

## Scoring

### Command Line Tool

The easiest way to score (using our pretrained models) is via our CLI tool, `emonet`.  The syntax for
this tool is:

```bash
emonet <emotion> <file to score>
```

Example:

```bash
emonet anger /Users/christophersantiago/emonet-data/wavs/6529_53113_1602547200.wav
```

**Note**: This CLI tool will run VAD on the `.wav` file, and can accept arbitrary length-- despite
model being trained on 8-second chunks. Therefore, you should use an original `.wav` of the sample
you wish to score, **not** a `.wav` that's been preprocessed with VAD.

### Python

You can also score via Python:

```python
from emonet import DATA_DIR, ROOT
from emonet.model import EmotionRegressor

def get_saved(emotion: str):
    return ROOT.joinpath('saved_models', f'{emotion}.ckpt')


emotion = 'fear'
model = EmotionRegressor.load_from_checkpoint(get_saved(emotion))

file = 'path-to-my-file'

model.score_file(file=file, sample_rate=16000, vad=True)
```

See `inference.py` for an example of how we scored our testing sets.
