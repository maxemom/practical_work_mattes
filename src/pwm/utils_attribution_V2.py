from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class RawAttributionResult:
    """
    Container for BAUTEIL B output.

    raw_target shape contract:
    - (T_gen, L_total, D)
    """

    raw_target: torch.Tensor
    source_ids_debug: torch.Tensor
    target_ids_debug: torch.Tensor


def _switch_attr_method_if_supported(inseq_model: Any, method_name: str) -> None:
    if hasattr(inseq_model, "load_attribution_method"):
        inseq_model.load_attribution_method(method_name)


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
