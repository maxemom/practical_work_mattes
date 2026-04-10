from typing import Dict, Any
import torch
import numpy as np

from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis
from sklearn.decomposition import KernelPCA

from pwm.utils_dimred_V2 import reduce_raw_target


def _safe_k(flat: np.ndarray, requested_k: int) -> int:
    n_samples, n_features = flat.shape
    max_k = min(n_samples, n_features)
    k = min(int(requested_k), int(max_k))
    if k < requested_k:
        print(f"[DimRed] Requested n_components={requested_k}, but only {max_k} possible → using {k}")
    return k


def reduce_raw_targets_to_importance(
    raw_targets: torch.Tensor,
    method_name: str,
    method_params: Dict[str, Any] | None = None,
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Canonical experiment reducer.

    Input:
      raw_targets with shape (L_total, T_gen, D)

    Output:
      normalized importance map with shape (L_total, T_gen)

    Invalid future/self positions stay NaN. Valid positions are normalized
    column-wise so that each generated-token column sums to 1.
    """
    if raw_targets.ndim != 3:
        raise ValueError(f"raw_targets must be 3D, got {tuple(raw_targets.shape)}")

    method_key = (method_name or "").strip().lower()
    params = dict(method_params or {})

    if method_key == "baseline":
        reduced = torch.norm(torch.nan_to_num(raw_targets, nan=0.0, posinf=0.0, neginf=0.0), p=2, dim=-1)
        invalid_mask = torch.isnan(raw_targets).all(dim=-1)
        reduced = reduced.to(torch.float32)
        reduced[invalid_mask] = float("nan")
    else:
        reduced = reduce_raw_target(raw_targets, method_key, params, seed=seed).to(torch.float32)

    reduced = torch.abs(reduced)
    normalized = torch.full_like(reduced, fill_value=float("nan"))

    _, gen_len = reduced.shape
    for step_i in range(gen_len):
        column = reduced[:, step_i]
        valid_mask = ~torch.isnan(column)
        if not bool(valid_mask.any()):
            continue

        valid_values = column[valid_mask]
        total = float(valid_values.sum().item())
        if total <= 0.0:
            normalized[valid_mask, step_i] = 0.0
            continue
        normalized[valid_mask, step_i] = valid_values / total

    return normalized
