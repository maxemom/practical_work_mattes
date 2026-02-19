from itertools import product
from typing import Any, Dict, List
from pwm.utils_base import deep_merge, stable_hash
from pwm.typess import Run

def build_runs(base_cfg: Dict[str, Any], grid_cfg: Dict[str, Any]) -> List[Run]:
    models = grid_cfg["models"]
    datasets = grid_cfg["datasets"]
    attributions = grid_cfg["attribution_functions"]
    dimreds = grid_cfg["dimensionality_reduction_methods"]

    runs: List[Run] = []

    for m, d, a, r in product(models, datasets, attributions, dimreds):
        resolved = deep_merge(base_cfg, {
            "model": m,
            "dataset": d,
            "attribution": a,
            "dimred": r,
        })
        run_id = stable_hash(resolved)

        runs.append(Run(
            model=m,
            dataset=d,
            attribution=a,
            dimred=r,
            run_id=run_id,
            resolved=resolved
        ))
    return runs
