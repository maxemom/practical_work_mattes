from typing import Dict, Any
import torch


def aggregate_baseline(
    raw_target: torch.Tensor,
    resolved: Dict[str, Any],
) -> torch.Tensor:
    """
    Aggregates raw embedding-level attributions according to baseline config.

    raw_target: (T_gen, T_total, D_embed)
    returns:    (T_gen, T_total)
    """
    agg_cfg = resolved.get("aggregation", {})
    baseline = (agg_cfg.get("baseline", "L2Norm") or "L2Norm").lower()
    normalize = bool(agg_cfg.get("normalize", False))

    # ✅ IMPORTANT: remove NaNs/Infs from padding etc.
    x = torch.nan_to_num(raw_target, nan=0.0, posinf=0.0, neginf=0.0)

    if baseline in ["l2", "l2norm"]:
        importance = torch.norm(x, p=2, dim=-1)

    elif baseline in ["l1", "l1norm"]:
        importance = torch.norm(x, p=1, dim=-1)

    elif baseline in ["meanabs"]:
        importance = torch.mean(torch.abs(x), dim=-1)

    elif baseline in ["maxabs"]:
        importance = torch.max(torch.abs(x), dim=-1).values

    else:
        raise ValueError(f"Unknown baseline aggregation: {baseline}")

    # Optional normalization per generated step (row-wise)
    if normalize:
        denom = importance.sum(dim=-1, keepdim=True)
        # ✅ handle zeros AND NaNs safely
        denom = torch.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0)
        denom = torch.where(denom <= 0, torch.ones_like(denom), denom)
        importance = importance / denom

    return importance