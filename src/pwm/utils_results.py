from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import json
import csv

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(obj):
        obj = asdict(obj)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_softnorm_steps_csv(path: Path, res: Any) -> None:
    """
    res: SoftNormResult (or dict with same keys)
    Writes a row per evaluated target step.
    """
    if is_dataclass(res):
        res = asdict(res)

    rows = []
    n = len(res["target_pos"])
    for i in range(n):
        rows.append({
            "target_pos": res["target_pos"][i],
            "target_token_id": res["target_token_id"][i],
            "soft_ns": res["soft_ns"][i],
            "soft_nc": res["soft_nc"][i],
            "dP0": res["dP0"][i],
            "dPR": res["dPR"][i],
            "dPnotR": res["dPnotR"][i],
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "target_pos","target_token_id","soft_ns","soft_nc","dP0","dPR","dPnotR"
        ])
        writer.writeheader()
        writer.writerows(rows)