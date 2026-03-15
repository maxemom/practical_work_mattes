from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import csv
import json


def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def save_json_v2(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_serializable(obj)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_softnorm_steps_csv_v2(path: Path, obj: Any) -> None:
    payload = _to_serializable(obj)
    fieldnames = ["target_pos", "target_token_id", "soft_ns", "soft_nc", "dP0", "dPR", "dPnotR"]

    rows = []
    n = len(payload.get("target_pos", []))
    for i in range(n):
        rows.append(
            {
                "target_pos": payload["target_pos"][i],
                "target_token_id": payload["target_token_id"][i],
                "soft_ns": payload["soft_ns"][i],
                "soft_nc": payload["soft_nc"][i],
                "dP0": payload["dP0"][i],
                "dPR": payload["dPR"][i],
                "dPnotR": payload["dPnotR"][i],
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_baseline_result_v2(prompt_dir: Path, attr_tag: str, result: Any) -> None:
    save_json_v2(prompt_dir / f"{attr_tag}_baseline.json", result)
    save_softnorm_steps_csv_v2(prompt_dir / f"{attr_tag}_baseline_steps.csv", result)


def save_dimred_result_v2(prompt_dir: Path, attr_tag: str, dimred_tag: str, result: Any) -> None:
    save_json_v2(prompt_dir / f"{attr_tag}_dimred_{dimred_tag}.json", result)
    save_softnorm_steps_csv_v2(prompt_dir / f"{attr_tag}_dimred_{dimred_tag}_steps.csv", result)
