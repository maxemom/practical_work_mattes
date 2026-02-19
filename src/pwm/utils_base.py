from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dict b into dict a (recursively) and return a new dict."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def expand_params(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand one grid item into a list of concrete param dicts.

    Supports:
      - item["params"] : a single dict
      - item["grid"]   : dict of param -> list-of-values (cartesian product)
      - none           : {}
    """
    if "grid" in item and item["grid"] is not None:
        grid = item["grid"]
        keys = list(grid.keys())
        values_lists = [grid[k] for k in keys]
        combos = []
        for values in product(*values_lists):
            combos.append({k: v for k, v in zip(keys, values)})
        return combos

    if "params" in item and item["params"] is not None:
        return [dict(item["params"])]

    return [{}]

def stable_hash(obj: Any) -> str:
    """Stable hash for dict/list structures (for run_id)."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]