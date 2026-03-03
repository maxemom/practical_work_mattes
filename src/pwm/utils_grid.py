from __future__ import annotations

from itertools import product
from typing import Any, Dict, List

from pwm.utils_base import deep_merge, stable_hash
from pwm.typess import Run


def build_runs(base_cfg: Dict[str, Any], grid_cfg: Dict[str, Any]) -> List[Run]:
    """
    Build runs only over (model, dataset, attribution).

    Dimensionality reduction methods are NOT part of the cartesian product anymore.
    They should be processed later in an inner loop in run_grid.py.

    We still attach the list of dimred methods to `resolved` so downstream code
    can access them uniformly.
    """
    models = grid_cfg["models"]
    datasets = grid_cfg["datasets"]
    attributions = grid_cfg["attribution_functions"]

    # Keep the full list for later inner-loop usage
    dimreds = grid_cfg.get("dimensionality_reduction_methods", [])

    runs: List[Run] = []

    for m, d, a in product(models, datasets, attributions):
        resolved = deep_merge(
            base_cfg,
            {
                "model": m,
                "dataset": d,
                "attribution": a,
                # store list, not a single method
                "dimensionality_reduction_methods": dimreds,
            },
        )

        # run_id now identifies (model,dataset,attr, + other base_cfg settings), not dimred
        run_id = stable_hash(resolved)

        runs.append(
            Run(
                model=m,
                dataset=d,
                attribution=a,
                dimred={"name": "MULTI", "params": {}},  # keep field for compatibility
                run_id=run_id,
                resolved=resolved,
            )
        )

    return runs