from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class SoftNormResultV3:
    target_pos: List[int]
    target_token_id: List[int]
    soft_ns: List[float]
    soft_nc: List[float]
    dP0: List[float]
    dPR: List[float]
    dPnotR: List[float]
    kept_tokens_R: List[float]
    kept_tokens_notR: List[float]
    soft_ns_mean: float
    soft_nc_mean: float
    mean_kept_tokens_R: float
    mean_kept_tokens_notR: float
    num_mc_samples: int
    rationale_size_mode: str
    warnings: List[str]


def _hellinger_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return (1.0 / (2.0 ** 0.5)) * torch.linalg.norm(torch.sqrt(p) - torch.sqrt(q), ord=2)


@torch.no_grad()
def _get_next_token_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, -1, :]
    return F.softmax(logits, dim=-1).squeeze(0)


@torch.no_grad()
def _get_next_token_probs_from_embeds(model, inputs_embeds: torch.Tensor) -> torch.Tensor:
    emb_dtype = model.get_input_embeddings().weight.dtype
    inputs_embeds = inputs_embeds.to(dtype=emb_dtype)
    seq_len = int(inputs_embeds.shape[1])
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=inputs_embeds.device)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = out.logits[:, -1, :]
    return F.softmax(logits, dim=-1).squeeze(0)


def _zero_baseline_embedding(model, device: torch.device) -> torch.Tensor:
    emb_weight = model.get_input_embeddings().weight
    return torch.zeros(emb_weight.shape[1], device=device, dtype=emb_weight.dtype)


def _sanitize_importance_column(
    importance_map: torch.Tensor,
    target_pos: int,
    step_i: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    raw_col = importance_map[:target_pos, step_i].to(torch.float32)
    col = torch.nan_to_num(raw_col, nan=0.0, posinf=0.0, neginf=0.0)
    # DimRed outputs can be signed. For faithfulness metrics we need a
    # non-negative importance magnitude, not the arbitrary projection sign.
    col = torch.abs(col)

    total = col.sum()
    if float(total) <= eps:
        raise ValueError(
            f"Importance values sum to ~0 for step_i={step_i}, target_pos={target_pos}"
        )
    return col / total


def _effective_support_size(weights: torch.Tensor, eps: float = 1e-12) -> float:
    denom = torch.sum(weights * weights).item()
    if denom <= eps:
        return 1.0
    return 1.0 / denom


def _solve_scaled_probs(weights: torch.Tensor, target_sum: float, iters: int = 48) -> torch.Tensor:
    if target_sum <= 0.0:
        return torch.zeros_like(weights)
    if target_sum >= float(weights.shape[0]):
        return torch.ones_like(weights)

    lo = 0.0
    hi = max(1.0, target_sum / max(weights.max().item(), 1e-12))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        probs = torch.clamp(mid * weights, 0.0, 1.0)
        if probs.sum().item() < target_sum:
            lo = mid
        else:
            hi = mid
    return torch.clamp(hi * weights, 0.0, 1.0)


def _build_keep_probs(
    weights: torch.Tensor,
    rationale_size_mode: str = "effective_support",
    rationale_fraction: float = 0.2,
) -> torch.Tensor:
    ctx_len = int(weights.shape[0])

    if rationale_size_mode == "effective_support":
        target_keep = _effective_support_size(weights)
    elif rationale_size_mode == "fraction":
        target_keep = max(1.0, rationale_fraction * ctx_len)
    elif rationale_size_mode == "all_mass":
        # Included for debugging/ablation; this degenerates towards V2-style low retention.
        target_keep = 1.0
    else:
        raise ValueError(
            f"Unknown rationale_size_mode '{rationale_size_mode}'. "
            "Expected one of: effective_support, fraction, all_mass."
        )

    target_keep = min(float(ctx_len), max(1.0, float(target_keep)))
    return _solve_scaled_probs(weights, target_keep)


@torch.no_grad()
def _perturb_inputs_embeds_with_mask(
    model,
    input_ids: torch.Tensor,
    keep_mask: torch.Tensor,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be (1, L), got {tuple(input_ids.shape)}")

    emb_layer = model.get_input_embeddings()
    embeds = emb_layer(input_ids)
    emb_dtype = embeds.dtype
    emb_device = embeds.device
    baseline = _zero_baseline_embedding(model, emb_device).view(1, 1, -1)
    keep_mask = keep_mask.to(device=emb_device, dtype=emb_dtype).view(1, -1, 1)
    return keep_mask * embeds + (1.0 - keep_mask) * baseline


@torch.no_grad()
def _sampled_distribution_average(
    model,
    input_ids: torch.Tensor,
    keep_probs: torch.Tensor,
    num_samples: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    prob_sum = None
    for _ in range(num_samples):
        mask = torch.bernoulli(keep_probs, generator=generator)
        embeds = _perturb_inputs_embeds_with_mask(model, input_ids, mask)
        probs = _get_next_token_probs_from_embeds(model, embeds)
        if prob_sum is None:
            prob_sum = probs
        else:
            prob_sum = prob_sum + probs
    return prob_sum / float(num_samples)


def compute_soft_norm_metrics_v3(
    model,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    importance_map: torch.Tensor,
    *,
    seed: Optional[int] = 42,
    num_mc_samples: int = 8,
    rationale_size_mode: str = "all_mass",
    rationale_fraction: float = 0.2,
) -> SoftNormResultV3:
    """
    Reimplementation of soft sufficiency/comprehensiveness for run_grid_V2.

    Main differences to V2:
    - keep probabilities are rescaled so that the expected number of retained tokens
      is not forced to 1
    - perturbed distributions are averaged over multiple Bernoulli samples
    """
    device = next(model.parameters()).device
    model.eval()

    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got {tuple(input_ids.shape)}")
    if generated_ids.ndim != 1:
        raise ValueError(f"generated_ids must be 1D, got {tuple(generated_ids.shape)}")
    if importance_map.ndim != 2:
        raise ValueError(f"importance_map must be 2D, got {tuple(importance_map.shape)}")
    if num_mc_samples <= 0:
        raise ValueError(f"num_mc_samples must be > 0, got {num_mc_samples}")

    prompt_len = int(input_ids.shape[0])
    gen_len = int(generated_ids.shape[0])
    total_len = prompt_len + gen_len

    if tuple(importance_map.shape) != (total_len, gen_len):
        raise ValueError(
            f"importance_map must have shape ({total_len}, {gen_len}), got {tuple(importance_map.shape)}"
        )

    total_ids = torch.cat((input_ids, generated_ids), dim=0)

    target_pos_list: List[int] = []
    target_token_id_list: List[int] = []
    soft_ns_list: List[float] = []
    soft_nc_list: List[float] = []
    dP0_list: List[float] = []
    dPR_list: List[float] = []
    dPnotR_list: List[float] = []
    kept_tokens_R_list: List[float] = []
    kept_tokens_notR_list: List[float] = []
    warnings: List[str] = []

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    for target_pos in range(prompt_len, total_len):
        step_i = target_pos - prompt_len
        ctx_ids = total_ids[:target_pos].unsqueeze(0).to(device)
        raw_col = importance_map[:target_pos, step_i].to(torch.float32)
        if torch.any(torch.nan_to_num(raw_col, nan=0.0, posinf=0.0, neginf=0.0) < 0):
            warnings.append(f"signed_importance_detected_at_step:{step_i}")
        weights = _sanitize_importance_column(importance_map, target_pos=target_pos, step_i=step_i).to(device)

        keep_probs_R = _build_keep_probs(
            weights,
            rationale_size_mode=rationale_size_mode,
            rationale_fraction=rationale_fraction,
        )
        keep_probs_notR = 1.0 - keep_probs_R
        keep_probs_zero = torch.zeros_like(keep_probs_R)

        p_full = _get_next_token_probs(model, ctx_ids)
        p_R = _sampled_distribution_average(model, ctx_ids, keep_probs_R, num_mc_samples, generator)
        p_notR = _sampled_distribution_average(model, ctx_ids, keep_probs_notR, num_mc_samples, generator)
        p_0 = _sampled_distribution_average(model, ctx_ids, keep_probs_zero, num_mc_samples, generator)

        dP_R = float(_hellinger_distance(p_full, p_R).item()) # distance between predictions full context and only rationales
        dP_notR = float(_hellinger_distance(p_full, p_notR).item())
        dP0 = float(_hellinger_distance(p_full, p_0).item())

        if dP0 <= 0.0:
            soft_ns = 0.0
            soft_nc = 0.0
        else:
            soft_ns = max(0.0, (dP0 - dP_R) / dP0)
            soft_nc = max(0.0, dP_notR / dP0)

        target_pos_list.append(int(target_pos))
        target_token_id_list.append(int(total_ids[target_pos].item()))
        soft_ns_list.append(float(soft_ns))
        soft_nc_list.append(float(soft_nc))
        dP0_list.append(float(dP0))
        dPR_list.append(float(dP_R))
        dPnotR_list.append(float(dP_notR))
        kept_tokens_R_list.append(float(keep_probs_R.sum().item()))
        kept_tokens_notR_list.append(float(keep_probs_notR.sum().item()))

    soft_ns_mean = float(sum(soft_ns_list) / max(1, len(soft_ns_list)))
    soft_nc_mean = float(sum(soft_nc_list) / max(1, len(soft_nc_list)))
    mean_kept_tokens_R = float(sum(kept_tokens_R_list) / max(1, len(kept_tokens_R_list)))
    mean_kept_tokens_notR = float(sum(kept_tokens_notR_list) / max(1, len(kept_tokens_notR_list)))

    return SoftNormResultV3(
        target_pos=target_pos_list,
        target_token_id=target_token_id_list,
        soft_ns=soft_ns_list,
        soft_nc=soft_nc_list,
        dP0=dP0_list,
        dPR=dPR_list,
        dPnotR=dPnotR_list,
        kept_tokens_R=kept_tokens_R_list,
        kept_tokens_notR=kept_tokens_notR_list,
        soft_ns_mean=soft_ns_mean,
        soft_nc_mean=soft_nc_mean,
        mean_kept_tokens_R=mean_kept_tokens_R,
        mean_kept_tokens_notR=mean_kept_tokens_notR,
        num_mc_samples=int(num_mc_samples),
        rationale_size_mode=rationale_size_mode,
        warnings=sorted(set(warnings)),
    )
