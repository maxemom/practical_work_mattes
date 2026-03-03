from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json
import csv


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(obj):
        obj = asdict(obj)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_softnorm_steps_csv(path: Path, res: Any) -> None:
    if is_dataclass(res):
        res = asdict(res)

    fieldnames = ["target_pos", "target_token_id", "soft_ns", "soft_nc", "dP0", "dPR", "dPnotR"]

    n = len(res.get("target_pos", []))
    rows = []
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)