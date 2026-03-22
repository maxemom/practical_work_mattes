from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

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

def save_baseline_result_v2(prompt_dir: Path, attr_tag: str, result: Any) -> None:
    save_json_v2(prompt_dir / f"{attr_tag}_baseline.json", result)


def save_dimred_result_v2(prompt_dir: Path, attr_tag: str, dimred_tag: str, result: Any) -> None:
    save_json_v2(prompt_dir / f"{attr_tag}_dimred_{dimred_tag}.json", result)
