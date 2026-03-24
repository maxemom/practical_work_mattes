import torch
import numpy as np

from sklearn.decomposition import FactorAnalysis, FastICA, PCA, NMF, KernelPCA


def _normalize_method_name(method_name: str) -> str:
    return (method_name or "").strip().lower().replace("_", "").replace(" ", "")


def _with_seed_defaults(method_key: str, method_params: dict, seed: int | None) -> dict:
    params = dict(method_params or {})
    if seed is None or "random_state" in params:
        return params

    if method_key in {"factoranalysis", "ica", "nmf"}:
        params["random_state"] = seed
        return params

    if method_key == "pca" and str(params.get("svd_solver", "auto")).lower() == "randomized":
        params["random_state"] = seed
        return params

    if method_key == "kernelpca" and str(params.get("eigen_solver", "auto")).lower() == "randomized":
        params["random_state"] = seed
        return params

    return params


def reduce_raw_target(
    x: torch.Tensor,
    method_name: str,
    method_params: dict | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Reduziert einen Tensor von shape (total_len, gen_len, dim) auf (total_len, gen_len)
    mittels einer angegebenen Dimensionality-Reduction-Methode.

    NaN-Handling:
        - Eine Position (total_len, gen_len) darf vollständig NaN sein:
          -> bedeutet: dort existiert kein Token
          -> diese Position wird beim Fitten ignoriert
          -> im Output wieder als NaN gesetzt
        - Teilweise NaNs innerhalb eines Token-Vektors sind nicht erlaubt
          -> das deutet auf beschädigte Daten hin

    Unterstützte Methoden:
        - "FactorAnalysis"
        - "ICA"
        - "PCA"
        - "KernelPCA"
        - "NMF"
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

    if x.ndim != 3:
        raise ValueError(f"x must have shape (total_len, gen_len, dim), got {tuple(x.shape)}")

    method_key = _normalize_method_name(method_name)
    method_params = _with_seed_defaults(method_key, method_params or {}, seed)

    total_len, gen_len, dim = x.shape
    if dim <= 0:
        raise ValueError(f"Last dimension must be > 0, got {dim}")

    original_device = x.device
    original_dtype = x.dtype if x.dtype.is_floating_point else torch.float32

    x_cpu = x.detach().to(torch.float32).cpu()

    if torch.isinf(x_cpu).any():
        raise ValueError("Input tensor contains inf values. Please clean them before dimensionality reduction.")

    out = torch.full((total_len, gen_len), fill_value=torch.nan, dtype=original_dtype, device=original_device)

    for step_i in range(gen_len):
        x_step = x_cpu[:, step_i, :]

        # Ganze Zeile NaN => dort existiert fuer diesen Schritt kein Token.
        all_nan_mask = torch.isnan(x_step).all(dim=1)

        # Teilweise NaNs innerhalb eines Token-Vektors bleiben ein Fehler.
        any_nan_mask = torch.isnan(x_step).any(dim=1)
        partial_nan_mask = any_nan_mask & (~all_nan_mask)

        if partial_nan_mask.any():
            bad_count = int(partial_nan_mask.sum().item())
            raise ValueError(
                f"Found {bad_count} rows with partial NaNs inside token embeddings at step {step_i}. "
                "Expected either fully valid rows or fully NaN rows."
            )

        valid_mask = ~all_nan_mask
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            continue

        x_valid = x_step[valid_mask].numpy()

        if method_key == "factoranalysis":
            reducer = FactorAnalysis(**method_params)
            reduced = reducer.fit_transform(x_valid)

        elif method_key == "ica":
            reducer = FastICA(**method_params)
            reduced = reducer.fit_transform(x_valid)

        elif method_key == "pca":
            reducer = PCA(**method_params)
            reduced = reducer.fit_transform(x_valid)

        elif method_key == "kernelpca":
            reducer = KernelPCA(**method_params)
            reduced = reducer.fit_transform(x_valid)

        elif method_key == "nmf":
            reducer = NMF(**method_params)
            reduced = reducer.fit_transform(np.abs(x_valid))

        else:
            raise ValueError(
                f"Unknown method_name='{method_name}'. "
                f"Supported: FactorAnalysis, ICA, PCA, KernelPCA, NMF"
            )

        if reduced.ndim != 2:
            raise ValueError(f"Reduced output must be 2D, got shape {reduced.shape}")

        n_components = reduced.shape[1]
        if n_components > 1:
            reduced_scalar_valid = np.linalg.norm(reduced, ord=2, axis=1)
        else:
            reduced_scalar_valid = np.abs(reduced[:, 0])

        reduced_scalar_full = np.full((total_len,), np.nan, dtype=np.float32)
        reduced_scalar_full[valid_mask.numpy()] = reduced_scalar_valid.astype(np.float32)
        out[:, step_i] = torch.from_numpy(reduced_scalar_full).to(device=original_device, dtype=original_dtype)

    return out
