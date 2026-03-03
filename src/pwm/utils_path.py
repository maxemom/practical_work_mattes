from pathlib import Path
from typing import Any, Dict
import yaml
import re


def safe_name(text: str) -> str:
    text = text.strip()
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^\w\-\.]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def build_output_dir(
    outputs_root: Path,
    model_name: str,
    dataset_name: str,
    attribution_name: str,
) -> Path:
    model_slug = safe_name(model_name)
    dataset_slug = safe_name(dataset_name)
    attr_slug = safe_name(attribution_name)
    return outputs_root / model_slug / dataset_slug / attr_slug


def save_resolved_config(run_dir: Path, resolved: Dict[str, Any]) -> None:
    path = run_dir / "resolved_config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)