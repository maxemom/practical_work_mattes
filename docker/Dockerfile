FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

# Install python deps (no conda inside docker to keep it simple)
RUN pip install --no-cache-dir \
    torch transformers accelerate inseq \
    numpy pandas scikit-learn pyyaml matplotlib pytest pipdeptree

COPY . /workspace

# No entrypoint yet (you will add once cli.py exists)
CMD ["bash", "-lc", "echo 'Docker image built. Add a CMD when your code is ready.'"]
