from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import inseq


@dataclass
class AttributionBatch:
    prompt: str
    generated_text: str
    raw_target: torch.Tensor           # (T_gen, W, D)
    source_ids: torch.Tensor           # (L_in,)
    target_ids: torch.Tensor           # (T_gen,)
    generated_ids: torch.Tensor        # (L_total,)
    source_tokens: Optional[list[str]] = None
    target_tokens: Optional[list[str]] = None


def _ensure_pad_token(tok) -> None:
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = tok.eos_token


@torch.no_grad()
def _generate_with_mask(inseq_model: inseq.AttributionModel, prompt: str, resolved: Dict[str, Any]) -> str:
    """
    Generate text with HF model directly (with attention_mask), return generated_text (decoded continuation).
    """
    tok = inseq_model.tokenizer
    _ensure_pad_token(tok)

    hf_model = inseq_model.model
    device = next(hf_model.parameters()).device
    hf_model.eval()

    model_params = resolved.get("model", {}).get("params", {})
    gen_defaults = resolved.get("generation", {})

    max_new_tokens = int(model_params.get("max_new_tokens", gen_defaults.get("max_new_tokens", 10)))
    do_sample = bool(model_params.get("do_sample", gen_defaults.get("do_sample", False)))
    temperature = float(model_params.get("temperature", gen_defaults.get("temperature", 1.0)))
    top_p = float(model_params.get("top_p", gen_defaults.get("top_p", 1.0)))

    enc = tok(prompt, return_tensors="pt", padding=False, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    gen_ids = hf_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )

    # gen_ids: (1, L_total)
    full_text = tok.decode(gen_ids[0], skip_special_tokens=False)

    return full_text


def model_attribute(
    inseq_model: inseq.AttributionModel,
    prompt: str,
    resolved: Dict[str, Any],
) -> AttributionBatch:
    """
    1) Generate continuation using HF model with attention_mask (prevents warning).
    2) Run inseq attribution using generated_texts (no generation inside inseq).
    3) Return raw target attributions and token ids.
    """
    tok = inseq_model.tokenizer
    _ensure_pad_token(tok)

    # 1) external generation (masked)
    generated_text = _generate_with_mask(inseq_model, prompt, resolved)

    # 2) inseq attribution on fixed (prompt, generated_text)
    attribution_params = resolved.get("attribution", {}).get("params", {})

    out = inseq_model.attribute(
        input_texts=prompt,
        generated_texts=generated_text,
        show_progress=False,
        **attribution_params,
    )

    seq = out.sequence_attributions[0]

    raw_target = seq.target_attributions
    if not isinstance(raw_target, torch.Tensor):
        raw_target = torch.tensor(raw_target)
    raw_target = raw_target.detach().cpu()

    if raw_target.ndim != 3:
        raise ValueError(f"Expected raw target attributions 3D (T_gen, W, D), got {raw_target.shape}")

    # Token objects → ids
    src = seq.source
    tgt = seq.target

    source_ids = torch.tensor([t.id for t in src], dtype=torch.long)
    target_ids = torch.tensor([t.id for t in tgt], dtype=torch.long)

    # -------------------------
    # Fix overlap: sometimes seq.target already includes the source prefix
    # -------------------------
    target_contains_source = (
        len(target_ids) >= len(source_ids)
        and torch.all(target_ids[: len(source_ids)] == source_ids).item()
    )

    if target_contains_source:
        # target_ids already is the full sequence (prompt + generated)
        generated_ids = target_ids
        target_only_ids = target_ids[len(source_ids):]
    else:
        # target_ids is only the continuation
        generated_ids = torch.cat([source_ids, target_ids], dim=0)
        target_only_ids = target_ids
    
    L_in = int(source_ids.shape[0])
    L_total = int(generated_ids.shape[0])
    T_gen = L_total - L_in

    # raw_target can be either (T_gen, W, D) OR (L_total, W, D)
    if raw_target.shape[0] == L_total:
        # keep only rows that correspond to generated tokens
        raw_target = raw_target[L_in:, :, :]
    elif raw_target.shape[0] == T_gen:
        pass
    else:
        raise ValueError(
            f"Unexpected raw_target first dim: got {raw_target.shape[0]}, expected {T_gen} or {L_total} "
            f"(T_gen={T_gen}, L_total={L_total}, L_in={L_in})."
        )
    

    try:
        source_tokens = [getattr(t, "token", getattr(t, "text", str(t))) for t in src]
    except Exception:
        source_tokens = None

    try:
        target_tokens = [getattr(t, "token", getattr(t, "text", str(t))) for t in tgt]
    except Exception:
        target_tokens = None

    return AttributionBatch(
        prompt=prompt,
        generated_text=generated_text,
        raw_target=raw_target,
        source_ids=source_ids,
        target_ids=target_ids,
        generated_ids=generated_ids,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
    )