from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class SoftNormResult:
    # per evaluated step
    target_pos: List[int]
    target_token_id: List[int]
    soft_ns: List[float]
    soft_nc: List[float]
    dP0: List[float]
    dPR: List[float]
    dPnotR: List[float]
    # summary
    soft_ns_mean: float
    soft_nc_mean: float


def hellinger_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    p, q: shape (V,) probability distributions
    returns scalar tensor
    """
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return (1.0 / (2.0 ** 0.5)) * torch.linalg.norm(torch.sqrt(p) - torch.sqrt(q), ord=2)


@torch.no_grad()
def _get_next_token_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    input_ids: (1, L)
    returns probs over vocab: (V,)
    """
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, -1, :]  # (1, V)
    probs = F.softmax(logits, dim=-1).squeeze(0)  # (V,)
    return probs


@torch.no_grad()
def _get_next_token_probs_from_embeds(model, inputs_embeds: torch.Tensor) -> torch.Tensor:
    # ensure embed dtype matches model embedding dtype
    emb_dtype = model.get_input_embeddings().weight.dtype
    inputs_embeds = inputs_embeds.to(dtype=emb_dtype)

    L = inputs_embeds.shape[1]
    attention_mask = torch.ones((1, L), dtype=torch.long, device=inputs_embeds.device)

    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = out.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1).squeeze(0)
    return probs


def _weights_from_importance(scores: torch.Tensor) -> torch.Tensor:
    """
    scores: (L,) nonnegative importance (often sums to 1)
    returns w in [0,1] for soft-mixing (keep weights).
    """
    scores = torch.nan_to_num(scores, nan=0.0)
    smax = scores.max()
    if float(smax) <= 0.0:
        return torch.zeros_like(scores)
    w = scores / smax  # now in [0,1]
    return torch.clamp(w, 0.0, 1.0)


def _baseline_embedding(model) -> torch.Tensor:
    """
    Returns a single baseline embedding vector e_baseline: (D,)
    We use mean of embedding matrix (stable default).
    """
    emb = model.get_input_embeddings().weight  # (V, D)
    return emb.mean(dim=0)  # (D,)


def soft_perturb_inputs_embeds(
    model,
    input_ids: torch.Tensor,
    w_keep: torch.Tensor,
) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be (1,L), got {input_ids.shape}")

    L = int(input_ids.shape[1])
    if w_keep.shape != (L,):
        raise ValueError(f"w_keep must be (L,), got {w_keep.shape} for L={L}")

    emb_layer = model.get_input_embeddings()
    e = emb_layer(input_ids)  # (1, L, D) dtype = model dtype (often float16 on MPS)
    e_dtype = e.dtype

    e_base = _baseline_embedding(model).to(device=e.device, dtype=e_dtype)  # (D,)

    # >>> CRITICAL: cast weights to same dtype as embeddings <<<
    w = w_keep.to(device=e.device, dtype=e_dtype).view(1, L, 1)  # (1, L, 1)

    e_pert = w * e + (1.0 - w) * e_base.view(1, 1, -1)
    # ensure dtype stays consistent
    return e_pert.to(dtype=e_dtype)


def compute_soft_norm_metrics(
    model,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    importance_map: torch.Tensor,
    metric_stride: int = 1,
) -> SoftNormResult:
    """
    Compute Soft-NS and Soft-NC for a single prompt+generation.

    IMPORTANT:
    - importance_map width does NOT need to be L_total or L_total-1.
      It can also be only source-length (e.g. prompt tokens only).
      We will pad missing context positions with zeros per step.

    Args:
        model: HF causal LM (already on correct device)
        input_ids:     (L_in,) token ids of prompt
        generated_ids: (L_total,) token ids prompt+generated
        importance_map: (T_gen, W) scalar importance per (generated-step, context_pos_available)
                        Row i corresponds to target_pos = L_in + i.
                        W can be:
                          - L_total
                          - L_total-1
                          - L_in (prompt-only)
                          - any W <= L_total (we pad per step)
        metric_stride: evaluate every k-th generated token.

    Returns:
        SoftNormResult with per-step + mean.
    """
    device = next(model.parameters()).device
    model.eval()

    # Ensure shapes
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got {input_ids.shape}")
    if generated_ids.ndim != 1:
        raise ValueError(f"generated_ids must be 1D, got {generated_ids.shape}")
    if importance_map.ndim != 2:
        raise ValueError(f"importance_map must be 2D, got {importance_map.shape}")

    L_in = int(input_ids.shape[0])
    L_total = int(generated_ids.shape[0])
    T_gen = L_total - L_in

    if T_gen <= 0:
        raise ValueError(f"Need generated tokens: L_total={L_total}, L_in={L_in}")

    if importance_map.shape[0] != T_gen:
        raise ValueError(
            f"importance_map first dim must be T_gen={T_gen}, got {importance_map.shape[0]}"
        )

    W = int(importance_map.shape[1])
    if W <= 0:
        raise ValueError("importance_map has zero width.")
    if W > L_total:
        raise ValueError(f"importance_map width {W} cannot exceed L_total {L_total}")

    target_pos_list: List[int] = []
    target_token_id_list: List[int] = []
    soft_ns_list: List[float] = []
    soft_nc_list: List[float] = []
    dP0_list: List[float] = []
    dPR_list: List[float] = []
    dPnotR_list: List[float] = []

    # Iterate over generated positions
    for target_pos in range(L_in, L_total, metric_stride):
        step_i = target_pos - L_in  # 0..T_gen-1

        # Context is everything before target_pos
        ctx_ids = generated_ids[:target_pos].unsqueeze(0).to(device)  # (1, L_ctx)
        L_ctx = int(ctx_ids.shape[1])

        # Extract row and align to current context length
        row = importance_map[step_i].to(device)  # (W,)
        use_len = min(int(row.shape[0]), L_ctx)

        row_use = row[:use_len]
        row_use = torch.nan_to_num(row_use, nan=0.0)

        # Pad to L_ctx so weights match ctx_ids length
        if use_len < L_ctx:
            row_use = F.pad(row_use, (0, L_ctx - use_len), value=0.0)

        # Convert to keep-weights in [0,1]
        w_R = _weights_from_importance(row_use)  # keep rationale
        w_notR = 1.0 - w_R                       # remove rationale
        w_zero = torch.zeros_like(w_R)           # fully perturbed baseline

        # Full distribution
        p_full = _get_next_token_probs(model, ctx_ids)  # (V,)

        # Perturbed distributions
        emb_R = soft_perturb_inputs_embeds(model, ctx_ids, w_R)
        p_R = _get_next_token_probs_from_embeds(model, emb_R)

        emb_notR = soft_perturb_inputs_embeds(model, ctx_ids, w_notR)
        p_notR = _get_next_token_probs_from_embeds(model, emb_notR)

        emb_0 = soft_perturb_inputs_embeds(model, ctx_ids, w_zero)
        p_0 = _get_next_token_probs_from_embeds(model, emb_0)

        # Deltas via Hellinger
        dP_R = float(hellinger_distance(p_full, p_R).item())
        dP_notR = float(hellinger_distance(p_full, p_notR).item())
        dP0 = float(hellinger_distance(p_full, p_0).item())

        # Soft metrics
        if dP0 <= 0.0:
            soft_ns = 0.0
            soft_nc = 0.0
        else:
            soft_ns = max(0.0, (dP0 - dP_R) / dP0)
            soft_nc = (dP_notR / dP0)

        target_pos_list.append(int(target_pos))
        target_token_id_list.append(int(generated_ids[target_pos].item()))
        soft_ns_list.append(float(soft_ns))
        soft_nc_list.append(float(soft_nc))
        dP0_list.append(float(dP0))
        dPR_list.append(float(dP_R))
        dPnotR_list.append(float(dP_notR))

    # Means
    soft_ns_mean = float(sum(soft_ns_list) / max(1, len(soft_ns_list)))
    soft_nc_mean = float(sum(soft_nc_list) / max(1, len(soft_nc_list)))

    return SoftNormResult(
        target_pos=target_pos_list,
        target_token_id=target_token_id_list,
        soft_ns=soft_ns_list,
        soft_nc=soft_nc_list,
        dP0=dP0_list,
        dPR=dPR_list,
        dPnotR=dPnotR_list,
        soft_ns_mean=soft_ns_mean,
        soft_nc_mean=soft_nc_mean,
    )