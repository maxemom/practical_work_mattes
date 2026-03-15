from dataclasses import dataclass
from typing import List, Optional

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


def get_normalized_importance_column(
    importance_map: torch.Tensor,
    target_pos: int,
    step_i: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    importance_map: (L_total, T_gen)
    For a fixed generated step_i, extract the valid context part [:target_pos, step_i],
    replace NaNs with 0, and normalize to sum to 1.

    Returns:
        col_norm: (target_pos,)
    """
    col = importance_map[:target_pos, step_i].to(torch.float32)
    col = torch.nan_to_num(col, nan=0.0)

    s = col.sum()
    if s.abs() < eps:
        raise ValueError(
            f"Importance values sum to ~0 for step_i={step_i}, target_pos={target_pos}"
        )

    return col / s


def hellinger_distance(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    p, q: (V,) probability distributions
    """
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return (1.0 / (2.0 ** 0.5)) * torch.linalg.norm(torch.sqrt(p) - torch.sqrt(q), ord=2)


@torch.no_grad()
def _get_next_token_probs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """
    input_ids: (1, L)
    returns: (V,)
    """
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, -1, :]  # (1, V)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    return probs


@torch.no_grad()
def _get_next_token_probs_from_embeds(model, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """
    inputs_embeds: (1, L, D)
    returns: (V,)
    """
    emb_dtype = model.get_input_embeddings().weight.dtype
    inputs_embeds = inputs_embeds.to(dtype=emb_dtype)

    L = int(inputs_embeds.shape[1])
    attention_mask = torch.ones((1, L), dtype=torch.long, device=inputs_embeds.device)

    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = out.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1).squeeze(0)
    return probs


def _zero_baseline_embedding(model, device: torch.device) -> torch.Tensor:
    """
    Returns zero baseline embedding vector of shape (D,)
    """
    emb_weight = model.get_input_embeddings().weight
    return torch.zeros(
        emb_weight.shape[1],
        device=device,
        dtype=emb_weight.dtype,
    )


def _sample_bernoulli_mask(
    probs: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    probs: (L,) in [0,1]
    returns mask: (L,) in {0,1}, float dtype
    """
    probs = torch.clamp(probs, 0.0, 1.0)
    return torch.bernoulli(probs, generator=generator)


@torch.no_grad()
def perturb_inputs_embeds_with_bernoulli(
    model,
    input_ids: torch.Tensor,
    keep_probs: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Paper-near perturbation:
      e'_i = m_i * e_i + (1 - m_i) * e_base
    with m_i ~ Bernoulli(q_i)

    input_ids:   (1, L)
    keep_probs:  (L,) in [0,1]
    returns:     (1, L, D)
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be (1, L), got {tuple(input_ids.shape)}")

    L = int(input_ids.shape[1])
    if keep_probs.shape != (L,):
        raise ValueError(f"keep_probs must have shape ({L},), got {tuple(keep_probs.shape)}")

    emb_layer = model.get_input_embeddings()
    e = emb_layer(input_ids)  # (1, L, D)
    e_dtype = e.dtype
    e_device = e.device

    e_base = _zero_baseline_embedding(model, e_device)  # (D,)

    mask = _sample_bernoulli_mask(
        keep_probs.to(device=e_device, dtype=torch.float32),
        generator=generator,
    ).to(dtype=e_dtype, device=e_device)  # (L,)

    mask = mask.view(1, L, 1)  # (1, L, 1)
    e_pert = mask * e + (1.0 - mask) * e_base.view(1, 1, -1)
    return e_pert


def _importance_to_probs(scores: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Convert normalized importance scores to Bernoulli probabilities in [0,1].

    Since your column already sums to 1, values are typically already in [0,1].
    We still clamp for safety.
    """
    scores = torch.nan_to_num(scores, nan=0.0)
    scores = torch.clamp(scores, min=0.0)

    s = scores.sum()
    if s.abs() < eps:
        return torch.zeros_like(scores)

    scores = scores / s
    return torch.clamp(scores, 0.0, 1.0)


def compute_soft_norm_metrics(
    model,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    importance_map: torch.Tensor,
    seed: Optional[int] = 42,
) -> SoftNormResult:
    """
    Paper-near Soft-NS / Soft-NC for a single prompt+generation.

    Args:
        model:
            HF causal LM already on correct device.
        input_ids:
            (L_in,) prompt token ids.
        generated_ids:
            (T_gen,) generated token ids only.
        importance_map:
            (L_total, T_gen), where L_total = L_in + T_gen.
            Column step_i contains importance scores for context positions.
            Structural invalid positions may be NaN.
        seed:
            random seed for Bernoulli masks.

    Returns:
        SoftNormResult
    """
    device = next(model.parameters()).device
    model.eval()

    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got {tuple(input_ids.shape)}")
    if generated_ids.ndim != 1:
        raise ValueError(f"generated_ids must be 1D, got {tuple(generated_ids.shape)}")
    if importance_map.ndim != 2:
        raise ValueError(f"importance_map must be 2D, got {tuple(importance_map.shape)}")

    L_in = int(input_ids.shape[0])
    T_gen = int(generated_ids.shape[0])
    L_total = L_in + T_gen

    if importance_map.shape != (L_total, T_gen):
        raise ValueError(
            f"importance_map must have shape ({L_total}, {T_gen}), "
            f"got {tuple(importance_map.shape)}"
        )

    total_ids = torch.cat((input_ids, generated_ids), dim=0)  # (L_total,)

    target_pos_list: List[int] = []
    target_token_id_list: List[int] = []
    soft_ns_list: List[float] = []
    soft_nc_list: List[float] = []
    dP0_list: List[float] = []
    dPR_list: List[float] = []
    dPnotR_list: List[float] = []

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    for target_pos in range(L_in, L_total):
        step_i = target_pos - L_in

        # context before current target token
        ctx_ids = total_ids[:target_pos].unsqueeze(0).to(device)  # (1, L_ctx)
        L_ctx = int(ctx_ids.shape[1])

        # normalized importance over current context
        importance_col = get_normalized_importance_column(
            importance_map=importance_map,
            target_pos=target_pos,
            step_i=step_i,
        ).to(device)  # (L_ctx,)
        

        if importance_col.shape != (L_ctx,):
            raise RuntimeError(
                f"importance_col has wrong shape {tuple(importance_col.shape)}, expected ({L_ctx},)"
            )

        # q = a_i  for Soft-Sufficiency / Soft-NS
        q_R = _importance_to_probs(importance_col)

        # q = 1 - a_i for Soft-Comprehensiveness / Soft-NC
        q_notR = torch.clamp(1.0 - q_R, 0.0, 1.0)

        # zero baseline => always drop
        q_zero = torch.zeros_like(q_R)

        # full distribution
        p_full = _get_next_token_probs(model, ctx_ids)

        # perturbed distributions
        emb_R = perturb_inputs_embeds_with_bernoulli(
            model=model,
            input_ids=ctx_ids,
            keep_probs=q_R,
            generator=generator,
        )
        p_R = _get_next_token_probs_from_embeds(model, emb_R)

        emb_notR = perturb_inputs_embeds_with_bernoulli(
            model=model,
            input_ids=ctx_ids,
            keep_probs=q_notR,
            generator=generator,
        )
        p_notR = _get_next_token_probs_from_embeds(model, emb_notR)

        emb_0 = perturb_inputs_embeds_with_bernoulli(
            model=model,
            input_ids=ctx_ids,
            keep_probs=q_zero,
            generator=generator,
        )
        p_0 = _get_next_token_probs_from_embeds(model, emb_0)

        # Hellinger deltas
        dP_R = float(hellinger_distance(p_full, p_R).item())
        dP_notR = float(hellinger_distance(p_full, p_notR).item())
        dP0 = float(hellinger_distance(p_full, p_0).item())

        if dP0 <= 0.0:
            soft_ns = 0.0
            soft_nc = 0.0
        else:
            soft_ns = max(0.0, (dP0 - dP_R) / dP0)
            soft_nc = dP_notR / dP0

        target_pos_list.append(int(target_pos))
        target_token_id_list.append(int(total_ids[target_pos].item()))
        soft_ns_list.append(float(soft_ns))
        soft_nc_list.append(float(soft_nc))
        dP0_list.append(float(dP0))
        dPR_list.append(float(dP_R))
        dPnotR_list.append(float(dP_notR))

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