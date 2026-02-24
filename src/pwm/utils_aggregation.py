from typing import Dict, Any
import torch


def aggregate_baseline(
    raw_target: torch.Tensor,
    resolved: Dict[str, Any],
) -> torch.Tensor:
    """
    Aggregates raw embedding-level attributions according to baseline config.

    Args:
        raw_target: Tensor of shape (T_gen, T_total, D_embed)
        resolved:   Full resolved config dict

    Returns:
        importance_map: Tensor of shape (T_gen, T_total)
    """

    agg_cfg = resolved.get("aggregation", {})
    baseline = agg_cfg.get("baseline", "L2Norm")
    normalize = agg_cfg.get("normalize", False)

    baseline = baseline.lower()

    if baseline in ["l2", "l2norm"]:
        importance = torch.norm(raw_target, p=2, dim=-1)

    elif baseline in ["l1", "l1norm"]:
        importance = torch.norm(raw_target, p=1, dim=-1)

    elif baseline in ["meanabs"]:
        importance = torch.mean(torch.abs(raw_target), dim=-1)

    elif baseline in ["maxabs"]:
        importance = torch.max(torch.abs(raw_target), dim=-1).values

    else:
        raise ValueError(f"Unknown baseline aggregation: {baseline}")

    # Optional normalization per target token
    if normalize:
        denom = importance.sum(dim=-1, keepdim=True)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        importance = importance / denom

    return importance