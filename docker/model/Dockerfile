FROM cjsantiago/emonet-base

WORKDIR /home/jovyan

COPY . /home/jovyan

USER root

RUN source activate && \
    conda activate emonet && \
    pip install -e .

RUN chmod -R 777 ./

ENTRYPOINT ["tini", "--"]
CMD ["/usr/local/bin/start.sh", "jupyter", "lab"]

USER $NB_USER
