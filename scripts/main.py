from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import create_plots
import prepare_dataset
import run_grid


def run_pipeline(
    *,
    base_path: str = "configs/base.yaml",
    grid_path: str = "configs/grid.yaml",
    max_prompts: int | None = None,
    skip_prepare: bool = False,
) -> None:
    if not skip_prepare:
        prepare_dataset.run_all_from_config(base_path)
    run_grid.run_grid(base_path=base_path, grid_path=grid_path, max_prompts=max_prompts)
    create_plots.create_all_plots(
        output_root="outputs",
        grid_path=grid_path,
        all_bar_dimreds=True,
        all_token_attributions=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline: prepare -> grid -> create_plots.")
    parser.add_argument("--base", type=str, default="configs/base.yaml")
    parser.add_argument("--grid", type=str, default="configs/grid.yaml")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        base_path=args.base,
        grid_path=args.grid,
        max_prompts=args.max_prompts,
        skip_prepare=args.skip_prepare,
    )


if __name__ == "__main__":
    main()
