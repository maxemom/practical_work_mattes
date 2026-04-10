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
