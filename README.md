# Practical Work – Attribution Functions for LLMs

This project evaluates attribution methods for decoder-only language models.  
The entire experimental pipeline is containerized using Docker to ensure reproducibility.

The repository contains small, processed evaluation datasets (100 samples each).  
Raw datasets are not included for size and licensing reasons.

---

## Requirements

- Docker
- Git

No local Python installation is required.

---

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd Practical_Work_Mattes
```

---

## 2. Build Docker Image

```bash
docker build -t pwm .
```

This builds a Docker image containing:

- Python 3.11
- PyTorch
- Transformers
- Inseq
- All required dependencies

The image contains **code and dependencies only** (no datasets).

---

## 3. Dataset Mounting

Processed evaluation datasets are included in the repository under:

```
data/processed/
```

Since the Docker image does not include data, the `data/` directory must be mounted into the container.

---

## 4. Run Experiments

Create an output directory:

```bash
mkdir -p outputs
```

Run the experiment pipeline:

```bash
docker run --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  pwm python scripts/run_grid.py --base configs/base.yaml --grid configs/grid.yaml
```

This will:

- Load prompts from `data/processed/`
- Generate model outputs
- Compute attribution scores
- Calculate evaluation metrics (e.g., comprehensiveness, sufficiency)
- Store results in `outputs/`

---

## 5. Generate Visualizations

After experiments have finished, create summary plots:

```bash
docker run --rm \
  -v "$PWD/outputs:/workspace/outputs" \
  pwm python scripts/create_visuals.py
```

Plots will be saved in:

```
outputs/plots/
```

---

## Project Structure

```
data/
  processed/        # small frozen evaluation datasets (committed)
  raw/              # optional, not tracked in git

outputs/            # experiment results (not tracked)

scripts/
  run_grid.py
  prepare_dataset.py
  create_visuals.py

configs/
  base.yaml
  grid.yaml

docker/
  Dockerfile
```

---

## Reproducibility

- Fixed random seed (see `configs/base.yaml`)
- Frozen evaluation datasets (100 samples per dataset)
- Fully containerized execution environment
- Results reproducible via Docker

---

## Notes

- Raw datasets are not included.
- Only processed evaluation subsets are committed.
- The Docker image intentionally excludes datasets and results.
