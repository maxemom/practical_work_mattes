from typing import Dict, Any
import torch
import numpy as np

from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis
from sklearn.decomposition import KernelPCA


def _safe_k(flat: np.ndarray, requested_k: int) -> int:
    n_samples, n_features = flat.shape
    max_k = min(n_samples, n_features)
    k = min(int(requested_k), int(max_k))
    if k < requested_k:
        print(f"[DimRed] Requested n_components={requested_k}, but only {max_k} possible → using {k}")
    return k


def aggregate_dimred(
    raw_target: torch.Tensor,
    resolved: Dict[str, Any],
    dimred_cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    raw_target: (T_gen, T_total, D_embed)
    returns:    (T_gen, T_total)
    """
    method = (dimred_cfg.get("name", "pca") or "pca").lower()
    params = dimred_cfg.get("params", {}) or {}
    requested_k = int(params.get("n_components", 1))
    seed = int(resolved.get("seeds", {}).get("seed", 42))

    T_gen, T_total, D = raw_target.shape

    flat = raw_target.reshape(-1, D).detach().cpu().numpy()
    flat = np.nan_to_num(flat, nan=0.0)

    k = _safe_k(flat, requested_k)

    if k < 1:
        print("[DimRed] Too few samples → fallback to L2 baseline")
        return torch.norm(raw_target, p=2, dim=-1)

    if method == "baseline":
        return torch.norm(raw_target, p=2, dim=-1)

    if method == "pca":
        reducer = PCA(n_components=k, svd_solver=str(params.get("svd_solver", "full")))

    elif method == "ica":
        reducer = FastICA(
            n_components=k,
            random_state=seed,
            max_iter=int(params.get("max_iter", 2000)),
            tol=float(params.get("tol", 1e-4)),
        )

    elif method == "nmf":
        # NMF requires non-negative
        flat_nonneg = np.abs(flat)
        reducer = NMF(
            n_components=k,
            init=str(params.get("init", "nndsvda")),
            max_iter=int(params.get("max_iter", 2000)),
            tol=float(params.get("tol", 1e-4)),
            random_state=seed,
        )
        flat = flat_nonneg

    elif method == "factor_analysis":
        reducer = FactorAnalysis(
            n_components=k,
            random_state=seed,
            max_iter=int(params.get("max_iter", 1000)),
            tol=float(params.get("tol", 1e-2)),
        )

    elif method == "kernel_pca":
        reducer = KernelPCA(
            n_components=k,
            kernel=str(params.get("kernel", "rbf")),
            gamma=float(params.get("gamma", 0.1)),
        )

    else:
        raise ValueError(f"Unknown dimred method: {method}")

    reduced = reducer.fit_transform(flat)

    reduced_tensor = torch.tensor(reduced, dtype=torch.float32).reshape(T_gen, T_total, k)

    if k > 1:
        importance = torch.norm(reduced_tensor, dim=-1)
    else:
        importance = reduced_tensor.squeeze(-1)

    return importance