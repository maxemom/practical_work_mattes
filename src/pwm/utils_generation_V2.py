from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pwm.utils_pipeline import set_global_seed


def _ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token


def _set_seed(seed: int) -> None:
    set_global_seed(int(seed))


def _stabilize_model_for_mps(model: Any) -> Any:
    """
    Best-effort MPS guard: keep model weights in float32 to avoid native
    mixed-dtype matmul assertion failures.
    """
    try:
        device = next(model.parameters()).device
    except Exception:
        return model

    if device.type == "mps":
        try:
            model.to(dtype=torch.float32)
        except Exception:
            pass
    return model


def load_generation_components(model_name: str, device: str) -> Tuple[Any, Any]:
    """
    Laedt HF-Model und Tokenizer fuer reine Generation.

    Returns:
    - model: HF causal LM (bereits auf device)
    - tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model = _stabilize_model_for_mps(model)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_output_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    generation_cfg: Dict[str, Any],
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """
    Fuehrt BAUTEIL A aus und gibt genau die benoetigten Outputs zurueck:
    - source_ids:    (L_in,)
    - generated_ids: (L_total,)
    - full_text:     str
    """
    _ensure_pad_token(tokenizer)
    _set_seed(seed)

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    max_new_tokens = int(generation_cfg.get("max_new_tokens", 10))
    temperature = float(generation_cfg.get("temperature", 1.0))
    top_p = float(generation_cfg.get("top_p", 1.0))
    do_sample = bool(generation_cfg.get("do_sample", temperature > 0.0))

    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=int(tokenizer.pad_token_id),
    )

    source_ids = input_ids[0].detach().cpu().to(torch.long)
    generated_ids = gen_ids[0].detach().cpu().to(torch.long)
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return source_ids, generated_ids, full_text
