FROM mambaorg/micromamba:1.5.8

# 1) Nur die Environment-Definition kopieren, damit Docker caching sauber funktioniert
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# 2) Conda-Env erstellen (das ist dein "Libraries-Layer")
RUN micromamba create -y -f /tmp/environment.yml && \
    micromamba clean --all --yes

# 3) Env im Container automatisch aktivieren
ENV MAMBA_DOCKERFILE_ACTIVATE=1

# 4) Projektcode kopieren (ändert sich häufig -> später in der Layer-Reihenfolge)
WORKDIR /workspace
COPY --chown=$MAMBA_USER:$MAMBA_USER . /workspace

# Default: interaktive Shell
CMD ["bash"]
