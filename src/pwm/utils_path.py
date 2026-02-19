from pathlib import Path
from typing import Any, Dict
import yaml
import re


def safe_name(text: str) -> str:
    """
    Replace problematic filesystem characters with '_'.
    Keeps names readable.
    """
    text = text.strip()
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^\w\-\.]", "_", text)
    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def params_to_suffix(params: Dict[str, Any]) -> str:
    if not params:
        return ""

    parts = [f"{k}={params[k]}" for k in sorted(params.keys())]
    return "__" + "__".join(parts)


def build_output_dir(
    outputs_root: Path,
    model_name: str,
    dataset_name: str,
    attribution_name: str,
    dimred_name: str,
    dimred_params: Dict[str, Any],
) -> Path:
    model_slug = safe_name(model_name)
    dataset_slug = safe_name(dataset_name)
    attr_slug = safe_name(attribution_name)

    dimred_slug = safe_name(dimred_name) + params_to_suffix(dimred_params)

    return outputs_root / model_slug / dataset_slug / attr_slug / dimred_slug

def save_resolved_config(run_dir: Path, resolved: Dict[str, Any]) -> None:
    path = run_dir / "resolved_config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)