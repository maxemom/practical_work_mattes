from typing import Dict, Any
import torch
import numpy as np

from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis
from sklearn.decomposition import KernelPCA


def aggregate_dimred(
    raw_target: torch.Tensor,
    resolved: Dict[str, Any],
) -> torch.Tensor:
    """
    Applies dimensionality reduction over embedding dimension
    and returns scalar importance map.

    raw_target shape:
        (T_gen, T_total, D_embed)

    Returns:
        importance_map shape:
        (T_gen, T_total)
    """

    dimred_cfg = resolved.get("dimred", {})
    method = dimred_cfg.get("name", "pca").lower()
    params = dimred_cfg.get("params", {})
    requested_k = int(params.get("n_components", 1))
    seed = int(resolved.get("seeds", {}).get("seed", 42))

    T_gen, T_total, D = raw_target.shape

    # Flatten
    flat = raw_target.reshape(-1, D).detach().cpu().numpy()

    # Replace NaNs (structural padding)
    flat = np.nan_to_num(flat, nan=0.0)

    n_samples = flat.shape[0]
    n_features = flat.shape[1]
    max_k = min(n_samples, n_features)

    # Cap n_components safely
    k = min(requested_k, max_k)

    if k < requested_k:
        print(
            f"[DimRed] Requested n_components={requested_k}, "
            f"but only {max_k} possible → using {k}"
        )

    # If too few samples → fallback to baseline L2
    if k < 1:
        print("[DimRed] Too few samples → fallback to L2 baseline")
        return torch.norm(raw_target, p=2, dim=-1)

    # Choose method
    if method == "pca":
        reducer = PCA(n_components=k, svd_solver="full")

    elif method == "ica":
        reducer = FastICA(n_components=k, random_state=seed)

    elif method == "nmf":
        flat = np.abs(flat)  # NMF requires non-negative
        reducer = NMF(n_components = k, init="random", max_iter=500, random_state=seed)

    elif method == "factor_analysis":
        reducer = FactorAnalysis(n_components=k, random_state=seed)

    elif method == "kernel_pca":
        reducer = KernelPCA(n_components=k, kernel="rbf")

    else:
        raise ValueError(f"Unknown dimred method: {method}")

    # Fit & transform
    reduced = reducer.fit_transform(flat)

    # Back to tensor
    reduced_tensor = (
        torch.tensor(reduced, dtype=torch.float32)
        .reshape(T_gen, T_total, k)
    )

    # Collapse to scalar importance
    if k > 1:
        importance = torch.norm(reduced_tensor, dim=-1)
    else:
        importance = reduced_tensor.squeeze(-1)

    return importance