FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch \
    torchvision

RUN python -m pip install \
    transformers \
    accelerate \
    inseq \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    pyyaml \
    matplotlib \
    pytest \
    pipdeptree \
    python-slugify \
    "git+https://github.com/niggoo/LRP-eXplains-Transformers.git"

COPY . /workspace

RUN python -m pip install -e .

CMD ["bash"]
