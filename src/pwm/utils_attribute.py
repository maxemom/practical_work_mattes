import torch
from typing import Any, Dict
import inseq


def model_attribute(
    inseq_model: inseq.AttributionModel,
    prompt: str,
    resolved: Dict[str, Any],
) -> torch.Tensor:
    """
    Runs attribution on a single prompt and returns raw target attributions.

    Returns:
        raw_target_attributions: Tensor of shape (T, L, D)
            T = number of generated tokens
            L = context length (input + prefix)
            D = embedding dimension
    """

    # -------------------------
    # 1. Generation parameters
    # -------------------------
    generation_cfg = resolved.get("generation", {})

    generation_args = {
        "max_new_tokens": generation_cfg.get("max_new_tokens", 10),
        "temperature": generation_cfg.get("temperature", 1.0),
        "top_p": generation_cfg.get("top_p", 1.0),
        "do_sample": generation_cfg.get("do_sample", False),
    }

    # -------------------------
    # 2. Attribution parameters
    # -------------------------
    attribution_params = resolved.get("attribution", {}).get("params", {})

    # -------------------------
    # 3. Run Inseq attribution
    # -------------------------
    out = inseq_model.attribute(
        input_texts=prompt,
        generation_args=generation_args,
        **attribution_params
    )

    # -------------------------
    # 4. Extract raw target attributions
    # -------------------------
    seq = out.sequence_attributions[0]
    raw_target = seq.target_attributions  # expected shape (T, L, D) or similar

    # Ensure torch tensor
    if not isinstance(raw_target, torch.Tensor):
        raw_target = torch.tensor(raw_target)

    # Optional: detach & move to CPU for safety
    raw_target = raw_target.detach().cpu()

    # -------------------------
    # 5. Sanity check
    # -------------------------
    if raw_target.ndim != 3:
        raise ValueError(
            f"Expected 3D target_attributions (T,L,D), got shape {raw_target.shape}"
        )

    return raw_target