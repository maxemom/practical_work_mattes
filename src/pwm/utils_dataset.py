from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def load_prompts(resolved: Dict[str, Any]) -> List[str]:
    """
    Returns a list of prompts (each prompt is one string) for iteration.

    Supported dataset config patterns:
      A) resolved["dataset"]["path"] = "data/prompts.txt" (one prompt per line)
      B) resolved["dataset"]["prompts"] = ["...", "..."] (inline prompts, good for tests)

    Optional:
      - resolved["dataset"]["params"]["max_samples"] limits number of prompts.
      - resolved["dataset"]["params"]["skip_empty"] defaults True.
    """
    dataset_cfg = resolved.get("dataset", {})
    params = dataset_cfg.get("params", {}) or {}
    max_samples = params.get("max_samples", None)
    skip_empty = params.get("skip_empty", True)

    # Inline prompts
    if "prompts" in dataset_cfg and dataset_cfg["prompts"] is not None:
        prompts = list(dataset_cfg["prompts"])
    else:
        # File-based prompts
        path = dataset_cfg.get("path")
        if not path:
            raise ValueError("Dataset config must contain either 'prompts' or 'path'.")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")

        prompts = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip("\n").strip()
                if skip_empty and not s:
                    continue
                prompts.append(s)

    if max_samples is not None:
        prompts = prompts[: int(max_samples)]

    if not prompts:
        raise ValueError("No prompts loaded (dataset empty after filtering).")

    return prompts