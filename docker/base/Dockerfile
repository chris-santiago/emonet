FROM jupyter/minimal-notebook:python-3.9.13

WORKDIR /home/jovyan

COPY env-base-plus.yml /home/jovyan

USER root

RUN apt-get update --yes && \
    apt-get install ffmpeg --yes

RUN source activate && \
    mamba env create -f env-base-plus.yml && \
    conda activate emonet && \
    conda install ipykernel -y && \
    python -m ipykernel install --name emonet --user

ENTRYPOINT ["tini", "--"]
CMD ["/usr/local/bin/start.sh", "jupyter", "lab"]

USER $NB_USER
