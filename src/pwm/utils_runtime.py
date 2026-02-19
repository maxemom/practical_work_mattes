from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


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
    names = []
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

    # Helper: validate an explicit cuda:N
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

    elif requested_norm in ("cpu",):
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
        # Unknown string -> fallback
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


def apply_device_resolution(resolved: Dict[str, Any], verbose: bool = True) -> DeviceReport:
    resolved.setdefault("runtime", {})
    requested = resolved["runtime"].get("device", "auto")

    report = resolve_device(requested)

    # write back
    resolved["runtime"]["device"] = report.chosen

    if verbose:
        print("=== Device Check ===")
        print(f"Requested: {report.requested}")
        print(f"Chosen:    {report.chosen}")
        if report.changed:
            print(f"Changed:   yes")
            if report.reason:
                print(f"Reason:    {report.reason}")
        else:
            print("Changed:   no")
        print(f"CUDA:      {report.cuda_available} (count={report.cuda_device_count})")
        if report.cuda_device_names:
            print(f"CUDA GPUs: {list(report.cuda_device_names)}")
        print(f"MPS:       {report.mps_available}")
        print("====================")

    return report