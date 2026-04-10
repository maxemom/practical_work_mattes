from __future__ import annotations

import json
import os
import random
import re
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ALLOWED_ATTR_METHODS = {
    "saliency",
    "input_x_gradient",
    "integrated_gradients",
    "deeplift",
    "gradient_shap",
    "lxt",
    "value_zeroing",
    "occlusion",
    "lime",
    "attention",
}

ALLOWED_DIMRED_METHODS = {
    "baseline",
    "pca",
    "ica",
    "factor_analysis",
    "nmf",
    "kernel_pca",
}


def disable_loading_verbosity() -> None:
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass


def configure_runtime_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Setting forward, backward hooks and attributes on non-linear",
        category=UserWarning,
    )


def patch_lxt_transformers_compatibility() -> None:
    """
    Keep optional LXT imports working on newer transformers versions where some
    older compatibility symbols were removed.
    """
    try:
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        return

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(
            heads: Any,
            n_heads: int,
            head_size: int,
            already_pruned_heads: Any,
        ) -> tuple[set[int], torch.Tensor]:
            pruned_heads = {int(head) for head in already_pruned_heads}
            remaining_heads = {int(head) for head in heads} - pruned_heads
            kept_indices = [
                idx
                for head in range(int(n_heads))
                if head not in remaining_heads
                for idx in range(head * int(head_size), (head + 1) * int(head_size))
            ]
            return remaining_heads, torch.tensor(kept_indices, dtype=torch.long)

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    try:
        from transformers.models.roberta import modeling_roberta
    except Exception:
        return

    if (
        not hasattr(modeling_roberta, "RobertaSdpaSelfAttention")
        and hasattr(modeling_roberta, "RobertaSelfAttention")
    ):
        modeling_roberta.RobertaSdpaSelfAttention = modeling_roberta.RobertaSelfAttention


def patch_lxt_attention_interface_compatibility() -> None:
    """
    LXT expects ALL_ATTENTION_FUNCTIONS to be a plain dict, while newer
    transformers expose an AttentionInterface object with get_interface().
    """
    try:
        from lxt.efficient import patches as lxt_patches
    except Exception:
        return

    if getattr(lxt_patches, "_pwm_attention_interface_patch", False):
        return

    original_patch_attention = lxt_patches.patch_attention
    original_patch_cp_attention = lxt_patches.patch_cp_attention

    def _patch_attention_collection(module: Any, wrap_fn: Any) -> bool:
        new_forward = wrap_fn(module.eager_attention_forward)
        if lxt_patches.check_already_patched(module.eager_attention_forward, new_forward):
            return False
        module.eager_attention_forward = new_forward

        attention_functions = getattr(module, "ALL_ATTENTION_FUNCTIONS", None)
        if attention_functions is None:
            return True

        patched_functions: Dict[str, Any] = {}
        for key, value in list(attention_functions.items()):
            new_value = wrap_fn(value)
            if lxt_patches.check_already_patched(value, new_value):
                return False
            patched_functions[key] = new_value

        if hasattr(attention_functions, "get_interface"):
            attention_functions.update(patched_functions)
        else:
            module.ALL_ATTENTION_FUNCTIONS = patched_functions
        return True

    def patch_attention(module: Any) -> bool:
        return _patch_attention_collection(module, lxt_patches.wrap_attention_forward)

    def patch_cp_attention(module: Any) -> bool:
        return _patch_attention_collection(module, lxt_patches.cp_wrap_attention_forward)

    lxt_patches.patch_attention = patch_attention
    lxt_patches.patch_cp_attention = patch_cp_attention

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("lxt.efficient.models."):
            continue
        for mapping_name, old_fn, new_fn in (
            ("attnLRP", original_patch_attention, patch_attention),
            ("cp_LRP", original_patch_cp_attention, patch_cp_attention),
        ):
            patch_map = getattr(module, mapping_name, None)
            if not isinstance(patch_map, dict):
                continue
            for key, value in list(patch_map.items()):
                if value is old_fn:
                    patch_map[key] = new_fn

    lxt_patches._pwm_attention_interface_patch = True


def patch_inseq_value_zeroing_tensor_output() -> None:
    """
    Inseq's ValueZeroing hook assumes tuple outputs, but recent decoder blocks
    may return a bare Tensor. Support both shapes.
    """
    try:
        from inseq.attr.feat.ops.value_zeroing import ValueZeroing
    except Exception:
        return

    if getattr(ValueZeroing, "_pwm_tensor_output_patch", False):
        return

    def get_states_extract_and_patch_hook(self, block_idx: int, hidden_state_idx: int = 0) -> Any:
        def states_extract_and_patch_forward_hook(module: Any, args: Any, output: Any) -> Any:
            del module, args
            if isinstance(output, torch.Tensor):
                self.corrupted_block_output_states[block_idx] = output.clone().float().detach().cpu()
                return self.clean_block_output_states[block_idx].to(output.device)

            hidden_state = output[hidden_state_idx]
            self.corrupted_block_output_states[block_idx] = hidden_state.clone().float().detach().cpu()
            clean_state = self.clean_block_output_states[block_idx].to(hidden_state.device)

            if isinstance(output, tuple):
                return output[:hidden_state_idx] + (clean_state,) + output[hidden_state_idx + 1 :]
            if isinstance(output, list):
                patched = list(output)
                patched[hidden_state_idx] = clean_state
                return patched
            return output

        return states_extract_and_patch_forward_hook

    ValueZeroing.get_states_extract_and_patch_hook = get_states_extract_and_patch_hook
    ValueZeroing._pwm_tensor_output_patch = True


def register_inseq_model_configs() -> None:
    try:
        import inseq
    except Exception:
        return

    patch_inseq_value_zeroing_tensor_output()

    registrations = [
        (
            "Gemma3ForCausalLM",
            {
                "self_attention_module": "self_attn",
                "value_vector": "value_states",
                "cross_attention_module": None,
            },
        ),
        (
            "Phi3ForCausalLM",
            {
                "self_attention_module": "self_attn",
                "value_vector": "value_states",
                "cross_attention_module": None,
            },
        ),
        (
            "MistralForCausalLM",
            {
                "self_attention_module": "self_attn",
                "value_vector": "value_states",
                "cross_attention_module": None,
            },
        ),
        (
            "Qwen3ForCausalLM",
            {
                "self_attention_module": "self_attn",
                "value_vector": "value_states",
                "cross_attention_module": None,
            },
        ),
        (
            "Qwen2ForCausalLM",
            {
                "self_attention_module": "self_attn",
                "value_vector": "value_states",
                "cross_attention_module": None,
            },
        ),
    ]

    for model_type, config in registrations:
        try:
            inseq.register_model_config(model_type=model_type, config=config, overwrite=True)
        except Exception:
            continue

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass


def safe_name(text: str) -> str:
    text = (text or "").strip().lower()
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^a-z0-9_\-\.]", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unknown"



def _tag_value(value: Any) -> str:
    return safe_name(str(value))


def _unique_tag(candidate: str, used: set[str]) -> str:
    if candidate not in used:
        used.add(candidate)
        return candidate

    suffix = 2
    while True:
        tagged = f"{candidate}_{suffix}"
        if tagged not in used:
            used.add(tagged)
            return tagged
        suffix += 1


def build_attr_index(attrs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    used: set[str] = set()
    for idx, cfg in enumerate(attrs):
        tag = _unique_tag(safe_name(cfg.get("name", "attr")), used)
        out[tag] = {"name": cfg.get("name"), "params": cfg.get("params", {}) or {}, "index": idx}
    return out


def build_dimred_index(dimreds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    used: set[str] = set()
    for idx, cfg in enumerate(dimreds):
        params = cfg.get("params", {}) or {}
        tag = safe_name(cfg.get("name", "dimred"))
        if "n_components" in params:
            tag = f"{tag}_n_components_{_tag_value(params['n_components'])}"
        tag = _unique_tag(tag, used)
        out[tag] = {"name": cfg.get("name"), "params": cfg.get("params", {}) or {}, "index": idx}
    return out


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def stabilize_model_for_metrics(model: Any) -> Any:
    """
    MPS safety: avoid half-precision matmul assertion crashes in metrics forward.
    """
    try:
        device = next(model.parameters()).device
    except Exception:
        return model

    if device.type == "mps":
        try:
            # Cast the full module, not only embeddings. Some MPS failures come
            # from later projection layers staying in bf16/fp16 while inputs are fp32.
            model.to(dtype=torch.float32)
        except Exception:
            # best-effort only; caller keeps running
            pass
    return model


def clear_device_cache(device: str) -> None:
    d = (device or "").lower()
    if d.startswith("cuda"):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return
    if d == "mps":
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
        return


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token


def load_prompts(dataset_cfg: Dict[str, Any], max_prompts: Optional[int]) -> List[str]:
    path = dataset_cfg.get("path")
    if not path:
        raise ValueError(f"Dataset '{dataset_cfg.get('name', 'unknown')}' is missing 'path'.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {p}")
    prompts: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                prompts.append(s)
    if max_prompts is not None:
        prompts = prompts[: max(0, int(max_prompts))]
    if not prompts:
        raise ValueError(f"No prompts loaded from dataset path: {p}")
    return prompts


@torch.no_grad()
def hf_generate_once(
    model: Any,
    tokenizer: Any,
    prompt: str,
    generation_cfg: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    ensure_pad_token(tokenizer)
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    gen_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 10)),
        do_sample=bool(generation_cfg.get("do_sample", False)),
        temperature=float(generation_cfg.get("temperature", 1.0)),
        top_p=float(generation_cfg.get("top_p", 1.0)),
        pad_token_id=int(tokenizer.pad_token_id),
    )
    # Keep these as regular tensors so later attribution/metric stages can use
    # them in autograd-tracked code paths without inference-mode restrictions.
    generated_ids = gen_ids[0].detach().cpu().long().clone()
    source_ids = input_ids[0].detach().cpu().long().clone()
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return source_ids, generated_ids, full_text



def importance_stats(x: torch.Tensor) -> Dict[str, float]:
    y = torch.nan_to_num(x.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
        "zeros_ratio": float((y == 0).float().mean().item()),
    }


def raw_target_nan_stats(raw_target: torch.Tensor, source_len: int) -> Dict[str, float]:
    """
    Returns:
      - global_nan_ratio: NaNs over full tensor (includes expected padded/future area)
      - active_nan_ratio: NaNs only over per-step active context width
    """
    if raw_target.ndim != 3:
        raise ValueError(f"raw_target must be 3D, got {tuple(raw_target.shape)}")

    t_gen, width, _ = raw_target.shape
    nan_mask = torch.isnan(raw_target)
    global_ratio = float(nan_mask.float().mean().item())

    active_nans = 0.0
    active_total = 0.0
    for i in range(t_gen):
        use_len = min(width, int(source_len) + i)
        if use_len <= 0:
            continue
        row_mask = nan_mask[i, :use_len, :]
        active_nans += float(row_mask.sum().item())
        active_total += float(row_mask.numel())

    if active_total <= 0:
        active_ratio = 0.0
    else:
        active_ratio = active_nans / active_total

    return {
        "global_nan_ratio": global_ratio,
        "active_nan_ratio": float(active_ratio),
    }


def append_error(prompt_dir: Path, err: Dict[str, Any]) -> None:
    path = prompt_dir / "error.json"
    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                old = json.load(f)
            if isinstance(old, list):
                existing = old
        except Exception:
            existing = []
    existing.append(err)
    with path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)




def switch_attr_method_if_supported(inseq_model: Any, method_name: str) -> None:
    if hasattr(inseq_model, "load_attribution_method"):
        inseq_model.load_attribution_method(method_name)
        return
    # No-op if runtime switching is unavailable; caller can provide a model
    # instance that was loaded with the desired attribution method.
    return
