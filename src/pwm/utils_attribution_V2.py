from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Dict

import torch


@dataclass
class RawAttributionResult:
    """
    Container for BAUTEIL B output.

    raw_target shape contract:
    - (L_total, T_gen, D)
    """

    raw_target: torch.Tensor
    source_ids_debug: torch.Tensor
    target_ids_debug: torch.Tensor


def _switch_attr_method_if_supported(inseq_model: Any, method_name: str) -> None:
    if hasattr(inseq_model, "load_attribution_method"):
        inseq_model.load_attribution_method(method_name)


def _prepare_model_for_lxt(model: Any) -> None:
    if getattr(model, "_pwm_lxt_prepared", False):
        return

    try:
        from lxt.efficient import monkey_patch
    except ImportError as exc:
        raise ImportError(
            "Custom LXT attribution requested, but the 'lxt' package is not installed."
        ) from exc

    model_module = importlib.import_module(model.__module__)
    monkey_patch(model_module, verbose=False)

    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    model.eval()
    setattr(model, "_pwm_lxt_prepared", True)


def _decode_generated_text(tokenizer: Any, generated_ids: torch.Tensor) -> str:
    ids = generated_ids.detach().cpu().to(torch.long)
    return tokenizer.decode(ids, skip_special_tokens=False)


def get_raw_targets_v2(
    inseq_model: Any,
    prompt: str,
    generated_ids: torch.Tensor,
    source_len: int,
    attr_name: str,
    attr_params: Dict[str, Any] | None = None,
) -> RawAttributionResult:
    """
    BAUTEIL B API:
    Input:
    - original prompt (str)
    - generated_ids (L_total,)
    - source_len (L_in)
    - attribution method + optional hyperparameters

    Output:
    - raw_target in fixed shape (T_gen, L_total, D), no aggregation.
    """
    if generated_ids.ndim != 1:
        raise ValueError(f"generated_ids must be 1D, got shape={tuple(generated_ids.shape)}")
    if int(source_len) <= 0:
        raise ValueError(f"source_len must be > 0, got {source_len}")
    if int(generated_ids.shape[0]) <= int(source_len):
        raise ValueError(
            f"generated_ids must be longer than source_len, got {generated_ids.shape[0]} <= {source_len}"
        )

    _switch_attr_method_if_supported(inseq_model, attr_name)
    generated_text = _decode_generated_text(inseq_model.tokenizer, generated_ids)

    out = inseq_model.attribute(
        input_texts=prompt,
        generated_texts=generated_text,
        show_progress=False,
        **(attr_params or {}),
    )
    seq = out.sequence_attributions[0]
    raw_target = seq.target_attributions
    if not isinstance(raw_target, torch.Tensor):
        raw_target = torch.tensor(raw_target)

    raw_target = raw_target.detach().cpu()

    source_ids_debug = torch.tensor([tok.id for tok in seq.source], dtype=torch.long)
    target_ids_debug = torch.tensor([tok.id for tok in seq.target], dtype=torch.long)

    del out
    del generated_text

    return RawAttributionResult(
        raw_target=raw_target,
        source_ids_debug=source_ids_debug,
        target_ids_debug=target_ids_debug,
    )


def get_raw_targets_lxt_v2(
    model: Any,
    generated_ids: torch.Tensor,
    source_len: int,
    attr_params: Dict[str, Any] | None = None,
) -> RawAttributionResult:
    """
    Custom LXT raw-target path with the same shape contract as the rest of the
    pipeline: (L_total, T_gen, D).
    """
    del attr_params  # reserved for future custom LXT options

    if generated_ids.ndim != 1:
        raise ValueError(f"generated_ids must be 1D, got shape={tuple(generated_ids.shape)}")
    if int(source_len) <= 0:
        raise ValueError(f"source_len must be > 0, got {source_len}")
    if int(generated_ids.shape[0]) <= int(source_len):
        raise ValueError(
            f"generated_ids must be longer than source_len, got {generated_ids.shape[0]} <= {source_len}"
        )

    _prepare_model_for_lxt(model)

    device = next(model.parameters()).device
    ids_cpu = generated_ids.detach().cpu().to(torch.long)
    total_len = int(ids_cpu.shape[0])
    t_gen = total_len - int(source_len)
    emb_dim = int(model.get_input_embeddings().weight.shape[1])

    raw_target = torch.full((total_len, t_gen, emb_dim), float("nan"), dtype=torch.float32)

    for step_i in range(t_gen):
        prefix_len = int(source_len) + step_i
        prefix_ids = ids_cpu[:prefix_len].to(device)
        target_token_id = int(ids_cpu[prefix_len].item())

        if hasattr(model, "zero_grad"):
            model.zero_grad(set_to_none=True)

        input_embeds = model.get_input_embeddings()(prefix_ids.unsqueeze(0))
        input_embeds = input_embeds.detach().requires_grad_(True)
        out = model(inputs_embeds=input_embeds, use_cache=False)
        target_logit = out.logits[0, -1, target_token_id]
        target_logit.backward()

        attr = (input_embeds.grad * input_embeds).detach().cpu().squeeze(0).to(torch.float32)
        raw_target[:prefix_len, step_i, :] = attr

    return RawAttributionResult(
        raw_target=raw_target,
        source_ids_debug=ids_cpu[:source_len].clone(),
        target_ids_debug=ids_cpu.clone(),
    )
