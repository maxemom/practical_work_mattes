from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List

import torch
import torch.nn.functional as F


@dataclass
class StrictMetricResult:
    target_pos: List[int]
    target_token_id: List[int]
    soft_ns_per_token: List[float]
    soft_nc_per_token: List[float]
    random_soft_ns_per_token: List[float]
    random_soft_nc_per_token: List[float]
    final_sufficiency_per_token: List[float]
    final_comprehensiveness_per_token: List[float]
    dP0: List[float]
    dP_R: List[float]
    dP_notR: List[float]
    random_dP_R: List[float]
    random_dP_notR: List[float]
    soft_ns_mean: float
    soft_nc_mean: float
    final_sufficiency_mean: float
    final_comprehensiveness_mean: float
    warnings: List[str]


def _hellinger_distance(p: torch.Tensor, q: torch.Tensor, eps: float) -> float:
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    distance = (1.0 / (2.0**0.5)) * torch.linalg.norm(torch.sqrt(p) - torch.sqrt(q), ord=2)
    return float(distance.item())


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _seed_for(base_seed: int, step_i: int, stream_offset: int) -> int:
    return int(base_seed) + 10_000 * int(step_i) + int(stream_offset)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_log_ratio(numerator: float, denominator: float, eps: float) -> float:
    if numerator <= eps and denominator <= eps:
        return 0.0
    return float(math.log(max(numerator, eps) / max(denominator, eps)))


@torch.inference_mode()
def _next_token_probs_from_embeds(model, inputs_embeds: torch.Tensor) -> torch.Tensor:
    device = inputs_embeds.device
    seq_len = int(inputs_embeds.shape[1])
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = out.logits[:, -1, :].to(torch.float32)
    return F.softmax(logits, dim=-1).squeeze(0)


def _broadcast_mask(token_probs: torch.Tensor, emb_dim: int, seed: int) -> torch.Tensor:
    probs_cpu = token_probs.detach().to(device="cpu", dtype=torch.float32)
    random_values = torch.rand((probs_cpu.shape[0], emb_dim), generator=_make_generator(seed), dtype=torch.float32)
    return (random_values <= probs_cpu.unsqueeze(-1)).to(torch.float32)


def _compute_soft_metrics(dP0: float, dP_R: float, dP_notR: float, eps: float) -> tuple[float, float]:
    if dP0 <= eps:
        return 0.0, 0.0
    soft_ns = max(0.0, dP0 - dP_R) / dP0
    soft_nc = dP_notR / dP0
    return float(soft_ns), float(soft_nc)


def compute_strict_soft_metrics(
    model,
    total_ids: torch.Tensor,
    source_len: int,
    importance_scores: torch.Tensor,
    *,
    seed: int,
    eps: float = 1e-12,
) -> StrictMetricResult:
    if total_ids.ndim != 1:
        raise ValueError(f"total_ids must be 1D, got {tuple(total_ids.shape)}")
    if importance_scores.ndim != 2:
        raise ValueError(f"importance_scores must be 2D, got {tuple(importance_scores.shape)}")

    device = next(model.parameters()).device
    model.eval()
    emb_layer = model.get_input_embeddings()
    emb_dtype = emb_layer.weight.dtype if device.type != "mps" else torch.float32

    total_ids = total_ids.detach().cpu().to(torch.long)
    importance_scores = torch.nan_to_num(
        importance_scores.detach().cpu().to(torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    total_len = int(total_ids.shape[0])
    source_len = int(source_len)
    t_gen = total_len - source_len
    if t_gen <= 0:
        raise ValueError("No generated tokens available for metric computation.")
    if tuple(importance_scores.shape) != (total_len, t_gen):
        raise ValueError(
            f"importance_scores must have shape ({total_len}, {t_gen}), got {tuple(importance_scores.shape)}"
        )

    target_pos_list: List[int] = []
    target_token_id_list: List[int] = []
    soft_ns_list: List[float] = []
    soft_nc_list: List[float] = []
    random_soft_ns_list: List[float] = []
    random_soft_nc_list: List[float] = []
    final_sufficiency_list: List[float] = []
    final_comprehensiveness_list: List[float] = []
    dP0_list: List[float] = []
    dP_R_list: List[float] = []
    dP_notR_list: List[float] = []
    random_dP_R_list: List[float] = []
    random_dP_notR_list: List[float] = []
    warnings: List[str] = []

    for step_i in range(t_gen):
        target_pos = source_len + step_i
        ctx_ids = total_ids[:target_pos].unsqueeze(0).to(device)
        original_embeds = emb_layer(ctx_ids).to(dtype=emb_dtype)
        ctx_len = int(original_embeds.shape[1])
        emb_dim = int(original_embeds.shape[2])

        scores = torch.abs(importance_scores[:target_pos, step_i])
        mass = float(scores.sum().item())
        if mass > eps:
            scores = scores / mass
        else:
            scores = torch.zeros_like(scores)
            warnings.append(f"zero_importance_mass_step:{step_i}")

        random_logits = torch.rand((ctx_len,), generator=_make_generator(_seed_for(seed, step_i, 11)))
        random_scores = F.softmax(random_logits, dim=0).to(torch.float32)

        mask_imp = _broadcast_mask(scores, emb_dim, _seed_for(seed, step_i, 21)).to(device=device, dtype=emb_dtype)
        mask_rand = _broadcast_mask(random_scores, emb_dim, _seed_for(seed, step_i, 31)).to(device=device, dtype=emb_dtype)

        embeds_imp = original_embeds * mask_imp.unsqueeze(0)
        embeds_not_imp = original_embeds * (1.0 - mask_imp.unsqueeze(0))
        embeds_zero = torch.zeros_like(original_embeds)
        embeds_rand = original_embeds * mask_rand.unsqueeze(0)
        embeds_not_rand = original_embeds * (1.0 - mask_rand.unsqueeze(0))

        p_full = _next_token_probs_from_embeds(model, original_embeds)
        p_imp = _next_token_probs_from_embeds(model, embeds_imp)
        p_not_imp = _next_token_probs_from_embeds(model, embeds_not_imp)
        p_zero = _next_token_probs_from_embeds(model, embeds_zero)
        p_rand = _next_token_probs_from_embeds(model, embeds_rand)
        p_not_rand = _next_token_probs_from_embeds(model, embeds_not_rand)

        dP0 = _hellinger_distance(p_full, p_zero, eps)
        dP_R = _hellinger_distance(p_full, p_imp, eps)
        dP_notR = _hellinger_distance(p_full, p_not_imp, eps)
        random_dP_R = _hellinger_distance(p_full, p_rand, eps)
        random_dP_notR = _hellinger_distance(p_full, p_not_rand, eps)

        soft_ns, soft_nc = _compute_soft_metrics(dP0, dP_R, dP_notR, eps)
        random_soft_ns, random_soft_nc = _compute_soft_metrics(dP0, random_dP_R, random_dP_notR, eps)
        final_sufficiency = _safe_log_ratio(soft_ns, random_soft_ns, eps)
        final_comprehensiveness = _safe_log_ratio(soft_nc, random_soft_nc, eps)

        if dP0 <= eps:
            warnings.append(f"degenerate_dP0_step:{step_i}")

        target_pos_list.append(int(target_pos))
        target_token_id_list.append(int(total_ids[target_pos].item()))
        soft_ns_list.append(soft_ns)
        soft_nc_list.append(soft_nc)
        random_soft_ns_list.append(random_soft_ns)
        random_soft_nc_list.append(random_soft_nc)
        final_sufficiency_list.append(final_sufficiency)
        final_comprehensiveness_list.append(final_comprehensiveness)
        dP0_list.append(dP0)
        dP_R_list.append(dP_R)
        dP_notR_list.append(dP_notR)
        random_dP_R_list.append(random_dP_R)
        random_dP_notR_list.append(random_dP_notR)

    return StrictMetricResult(
        target_pos=target_pos_list,
        target_token_id=target_token_id_list,
        soft_ns_per_token=soft_ns_list,
        soft_nc_per_token=soft_nc_list,
        random_soft_ns_per_token=random_soft_ns_list,
        random_soft_nc_per_token=random_soft_nc_list,
        final_sufficiency_per_token=final_sufficiency_list,
        final_comprehensiveness_per_token=final_comprehensiveness_list,
        dP0=dP0_list,
        dP_R=dP_R_list,
        dP_notR=dP_notR_list,
        random_dP_R=random_dP_R_list,
        random_dP_notR=random_dP_notR_list,
        soft_ns_mean=_mean(soft_ns_list),
        soft_nc_mean=_mean(soft_nc_list),
        final_sufficiency_mean=_mean(final_sufficiency_list),
        final_comprehensiveness_mean=_mean(final_comprehensiveness_list),
        warnings=sorted(set(warnings)),
    )
