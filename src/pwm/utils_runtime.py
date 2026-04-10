from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch

from pwm.utils_base import deep_merge


# -------------------------
# Device resolution
# -------------------------

@dataclass
class DeviceReport:
    requested: str
    chosen: str
    changed: bool
    reason: Optional[str]
    cuda_available: bool
    mps_available: bool
    cuda_device_count: int
    cuda_device_names: Tuple[str, ...]


def _normalize_device(device: str) -> str:
    d = (device or "").strip().lower()
    return d if d else "auto"


def _cuda_device_names() -> Tuple[str, ...]:
    if not torch.cuda.is_available():
        return tuple()
    names: List[str] = []
    for i in range(torch.cuda.device_count()):
        try:
            names.append(torch.cuda.get_device_name(i))
        except Exception:
            names.append(f"cuda:{i}")
    return tuple(names)


def resolve_device(requested: str) -> DeviceReport:
    requested_norm = _normalize_device(requested)

    cuda_avail = torch.cuda.is_available()
    mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_count = torch.cuda.device_count() if cuda_avail else 0
    cuda_names = _cuda_device_names()

    def cuda_index_ok(dev: str) -> bool:
        if not dev.startswith("cuda:"):
            return False
        try:
            idx = int(dev.split(":", 1)[1])
        except Exception:
            return False
        return cuda_avail and (0 <= idx < cuda_count)

    chosen = requested_norm
    reason = None

    if requested_norm == "auto":
        if cuda_avail:
            chosen = "cuda:0"
        elif mps_avail:
            chosen = "mps"
        else:
            chosen = "cpu"

    elif requested_norm == "cpu":
        chosen = "cpu"

    elif requested_norm in ("cuda", "cuda:0"):
        if cuda_avail:
            chosen = "cuda:0"
        else:
            chosen = "mps" if mps_avail else "cpu"
            reason = "CUDA requested but not available"

    elif requested_norm.startswith("cuda:"):
        if cuda_index_ok(requested_norm):
            chosen = requested_norm
        else:
            chosen = "cuda:0" if cuda_avail else ("mps" if mps_avail else "cpu")
            reason = f"Requested {requested_norm} not available (gpu count={cuda_count})"

    elif requested_norm == "mps":
        if mps_avail:
            chosen = "mps"
        else:
            chosen = "cuda:0" if cuda_avail else "cpu"
            reason = "MPS requested but not available"

    else:
        chosen = "cuda:0" if cuda_avail else ("mps" if mps_avail else "cpu")
        reason = f"Unknown device string '{requested_norm}'"

    return DeviceReport(
        requested=requested,
        chosen=chosen,
        changed=(requested_norm != chosen),
        reason=reason,
        cuda_available=cuda_avail,
        mps_available=mps_avail,
        cuda_device_count=cuda_count,
        cuda_device_names=cuda_names,
    )


# -------------------------
# Generation overrides (model.params -> generation)
# -------------------------

@dataclass
class GenerationReport:
    before: Dict[str, Any]
    after: Dict[str, Any]
    overridden: Dict[str, Dict[str, Any]]  # key -> {"from": x, "to": y}
    source: str  # where overrides came from (e.g. "model.params")


DTYPE_ALIASES: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


# Whitelist: only keys that should be treated as generation settings
GENERATION_KEYS = {
    "max_new_tokens",
    "min_new_tokens",
    "max_length",
    "min_length",
    "temperature",
    "top_p",
    "top_k",
    "do_sample",
    "num_beams",
    "repetition_penalty",
    "length_penalty",
    "no_repeat_ngram_size",
    "early_stopping",
    "eos_token_id",
    "pad_token_id",
    "bos_token_id",
}


def apply_generation_overrides(resolved: Dict[str, Any]) -> GenerationReport:
    """
    Merges generation defaults with per-model overrides.
    Convention:
      - resolved["generation"] contains defaults (base.yaml)
      - resolved["model"]["params"] can contain generation keys which override defaults
    Writes back to resolved["generation"] (final values).
    """
    resolved.setdefault("generation", {})
    resolved.setdefault("model", {})
    model_params = resolved.get("model", {}).get("params", {}) or {}

    before = dict(resolved["generation"])
    after = dict(resolved["generation"])
    overridden: Dict[str, Dict[str, Any]] = {}

    for k, v in model_params.items():
        if k in GENERATION_KEYS:
            old = after.get(k, None)
            after[k] = v
            if old != v:
                overridden[k] = {"from": old, "to": v}

    resolved["generation"] = after

    return GenerationReport(
        before=before,
        after=after,
        overridden=overridden,
        source="model.params",
    )


def resolve_model_dtype(resolved: Dict[str, Any]) -> tuple[Optional[torch.dtype], str]:
    runtime = resolved.get("runtime", {}) or {}
    model_params = resolved.get("model", {}).get("params", {}) or {}
    requested = model_params.get("dtype", runtime.get("dtype", "auto"))
    device = str(runtime.get("device", "cpu")).lower()

    if requested is None:
        requested = "auto"

    if isinstance(requested, torch.dtype):
        return requested, str(requested).replace("torch.", "")

    requested_name = str(requested).strip().lower()
    if requested_name != "auto":
        if requested_name not in DTYPE_ALIASES:
            raise ValueError(
                f"Unknown dtype '{requested}'. Supported: auto, {sorted(DTYPE_ALIASES.keys())}"
            )
        return DTYPE_ALIASES[requested_name], requested_name

    if device.startswith("cuda"):
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16, "bfloat16"
            return torch.float16, "float16"
        return torch.float32, "float32"

    if device == "mps":
        # MPS is prone to mixed-dtype matmul assertion failures with decoder
        # models and attribution forwards. Prefer fp32 by default for stability.
        return torch.float32, "float32"

    if device == "cpu":
        return torch.float32, "float32"

    return torch.float32, "float32"


def build_resolved_run_config(
    base_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    attrs: List[Dict[str, Any]],
    dimreds: List[Dict[str, Any]],
    chosen_device: str,
) -> Dict[str, Any]:
    """
    Build the final run config for run_grid_V2.

    - base_cfg["generation"] provides defaults
    - model_cfg["params"] may override generation keys via the whitelist above
    - the final generation config is materialized in resolved["generation"]
    """
    resolved = deep_merge(
        base_cfg,
        {
            "model": model_cfg,
            "dataset": dataset_cfg,
            "attribution_functions": attrs,
            "dimensionality_reduction_methods": dimreds,
            "runtime": {"device": chosen_device},
        },
    )
    apply_generation_overrides(resolved)
    return resolved


# -------------------------
# Combined runtime apply + report
# -------------------------

def apply_runtime_resolution(resolved: Dict[str, Any], verbose: bool = True) -> Tuple[DeviceReport, GenerationReport]:
    # device
    resolved.setdefault("runtime", {})
    requested = resolved["runtime"].get("device", "auto")
    device_report = resolve_device(requested)
    resolved["runtime"]["device"] = device_report.chosen

    # generation overrides
    gen_report = apply_generation_overrides(resolved)

    if verbose:
        print("=== Runtime Check ===")

        # Device part
        print("[Device]")
        print(f"  Requested: {device_report.requested}")
        print(f"  Chosen:    {device_report.chosen}")
        if device_report.changed:
            print(f"  Changed:   yes")
            if device_report.reason:
                print(f"  Reason:    {device_report.reason}")
        else:
            print("  Changed:   no")
        print(f"  CUDA:      {device_report.cuda_available} (count={device_report.cuda_device_count})")
        if device_report.cuda_device_names:
            print(f"  CUDA GPUs: {list(device_report.cuda_device_names)}")
        print(f"  MPS:       {device_report.mps_available}")

        # Generation part
        print("[Generation]")
        if gen_report.overridden:
            print(f"  Overrides from {gen_report.source}:")
            for k, d in gen_report.overridden.items():
                print(f"    - {k}: {d['from']} -> {d['to']}")
        else:
            print("  No per-model generation overrides applied.")

        print("  Final generation:")
        for k in sorted(gen_report.after.keys()):
            print(f"    {k}: {gen_report.after[k]}")

        print("====================")

    return device_report, gen_report
