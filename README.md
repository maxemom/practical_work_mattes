# Practical Work: Reproducible Attribution Pipeline

This repository runs a reproducible pipeline for decoder-only LLM attribution:

- HF generation (`model.generate` with `attention_mask`)
- inseq attribution on fixed `(prompt, generated_text)`
- baseline and dimred postprocessing from one raw attribution tensor
- SoftNorm metrics with A4 batching (`full`, `R`, `notR`, `0` in one forward)

## Run with Docker

Build image:

```bash
docker build -t pwm -f docker/Dockerfile .
```

Run pipeline (mount processed data only):

```bash
mkdir -p outputs
docker run --rm \
  -v "$PWD/data/processed:/workspace/data/processed:ro" \
  -v "$PWD/outputs:/workspace/outputs" \
  pwm python scripts/run_grid.py --base configs/base.yaml --grid configs/grid.yaml --device auto
```

Optional minimal/debug run:

```bash
docker run --rm \
  -v "$PWD/data/processed:/workspace/data/processed:ro" \
  -v "$PWD/outputs:/workspace/outputs" \
  pwm python scripts/run_grid.py \
    --base configs/base.yaml \
    --grid configs/grid.yaml \
    --device mps \
    --max-prompts 2 \
    --only-attr saliency \
    --only-dimred pca \
    --only-prompt-idx 0
```

`scripts/prepare_dataset.py` remains unchanged and can still be used to create/update files under `data/processed/`.

## Output Layout

Per `(model, dataset)` run:

```text
outputs/<model_slug>/<dataset_slug>/
  resolved_config.yaml
  attr_index.json
  dimred_index.json
  run_meta.json
  prompts/
    prompt_000/
      debug.json
      <aTag>_baseline.json
      <aTag>_baseline_steps.csv
      <aTag>_dimred_<dTag>.json
      <aTag>_dimred_<dTag>_steps.csv
      error.json            # only when failures happen
```

`aTag` / `dTag` are short stable tags; method params are stored in `attr_index.json` / `dimred_index.json`.

## Notes

- Raw data under `data/raw/` is kept for completeness and is not required for `run_grid.py`.
- The container image excludes datasets; processed prompts are expected to be mounted at runtime.
