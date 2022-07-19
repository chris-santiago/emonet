# Model Image

## Building

This `Dockerfile` takes the base image and adds model package.  From the **root directory**, run

```bash
docker build -t cjsantiago/emonet-model -f docker/model/Dockerfile .
```

to build.

## Running
```bash
docker run -it --name emonet -p 8888:8888 -v "${HOME}"/emonet-data:/home/jovyan/emonet-data  cjsantiago/emonet-model
```

## Scoring

You can score file(s) or signal(s), either on their own or with your own custom DataLoader, without the data directory (described below). 

## Training

**Important**: Training the model or running batch inference presumes that you have a `emonet-data` directory within your home folder, containing the original `voice_labeling_report` directory. This will allow you to replicate all batch preprocessing done prior to model training.

This minimal data directory should be structured in this way:

```bash
emonet-data

├── voice_labeling_report
  ├── voice_labels.json
  └── voice_samples
```
