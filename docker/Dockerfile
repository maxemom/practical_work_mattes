FROM mambaorg/micromamba:1.5.8

USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

RUN micromamba create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

ENV MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace

CMD ["bash"]
