#"""This script runs a grid over different models, datasets, attribution functions, and dimensionality reduction methods. Every combination is tested individually, and every combination result is stored with a Run_ID in outputs. Also, some overall results and tables are stored. This script is the core of the project."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List
from torch.serialization import load

from pwm.utils_model import prepare_inseq
from pwm.utils_grid import build_runs
from pwm.utils_path import build_output_dir, save_resolved_config
from pwm.utils_base import load_yaml
from pwm.utils_model import prepare_inseq
from pwm.utils_runtime import apply_runtime_resolution
from pwm.utils_dataset import load_prompts
from pwm.utils_attribute import model_attribute



# -------------------------
# Main execution
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="configs/base.yaml")
    parser.add_argument("--grid", type=str, default="configs/grid.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only print runs, do not execute.")
    args = parser.parse_args()

    base_cfg = load_yaml(Path(args.base))
    grid_cfg = load_yaml(Path(args.grid))

    # Small robustness defaults
    base_cfg.setdefault("paths", {})
    base_cfg["paths"].setdefault("data_dir", "data")
    base_cfg["paths"].setdefault("output_dir", "outputs")

    runs = build_runs(base_cfg, grid_cfg)
    print(f"Planned runs: {len(runs)}")

    for i, run in enumerate(runs, start=1):
        print(
        f"[{i}/{len(runs)}] run_id={run.run_id} | "
        f"model={run.model['name']} | dataset={run.dataset['name']} | "
        f"attr={run.attribution['name']} | dimred={run.dimred['name']} {run.dimred.get('params', {})}"
    )
        if args.dry_run:
            continue

        outputs_root = Path(base_cfg["paths"]["output_dir"])

        run_dir = build_output_dir(
            outputs_root=outputs_root,
            model_name=run.model["name"],
            dataset_name=run.dataset["name"],
            attribution_name=run.attribution["name"],
            dimred_name=run.dimred["name"],
            dimred_params=run.dimred.get("params", {}),
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        save_resolved_config(run_dir, run.resolved)
        apply_runtime_resolution(run.resolved, verbose=True)
        inseq_model = prepare_inseq(run.resolved)
        promts = load_prompts(run.resolved)
        for idx, prompt in enumerate(promts):
            raw_target = model_attribute(inseq_model, prompt=prompt, resolved=run.resolved)
            

if __name__ == "__main__":
    main()