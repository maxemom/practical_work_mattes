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


def name_prefix(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower()) or "x"


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


def resolve_device(requested: str) -> str:
    req = (requested or "auto").lower()
    cuda_ok = torch.cuda.is_available()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if req == "auto":
        if cuda_ok:
            return "cuda:0"
        if mps_ok:
            return "mps"
        return "cpu"
    if req == "mps":
        return "mps" if mps_ok else ("cuda:0" if cuda_ok else "cpu")
    if req.startswith("cuda"):
        return req if cuda_ok else ("mps" if mps_ok else "cpu")
    return "cpu"


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


def extract_raw_target_with_alignment(
    out: Any,
    source_ids: torch.Tensor,
    generated_ids: torch.Tensor,
) -> torch.Tensor:
    seq = out.sequence_attributions[0]
    raw_target = seq.target_attributions
    if not isinstance(raw_target, torch.Tensor):
        raw_target = torch.tensor(raw_target)
    raw_target = raw_target.detach().cpu()
    if raw_target.ndim != 3:
        raise ValueError(f"Expected raw_target ndim=3, got shape={tuple(raw_target.shape)}")

    src = seq.source
    tgt = seq.target
    src_seq_ids = torch.tensor([t.id for t in src], dtype=torch.long)
    tgt_seq_ids = torch.tensor([t.id for t in tgt], dtype=torch.long)

    contains_source = (
        len(tgt_seq_ids) >= len(src_seq_ids)
        and bool(torch.all(tgt_seq_ids[: len(src_seq_ids)] == src_seq_ids).item())
    )
    if contains_source:
        generated_from_inseq = tgt_seq_ids
    else:
        generated_from_inseq = torch.cat([src_seq_ids, tgt_seq_ids], dim=0)

    if generated_from_inseq.shape[0] != generated_ids.shape[0]:
        pass

    l_in = int(source_ids.shape[0])
    l_total = int(generated_ids.shape[0])
    t_gen = l_total - l_in
    if raw_target.shape[0] == l_total:
        raw_target = raw_target[l_in:, :, :]
    elif raw_target.shape[0] != t_gen:
        raise ValueError(
            f"Unexpected raw_target first dim={raw_target.shape[0]}, expected {t_gen} or {l_total}"
        )
    if raw_target.shape[0] != t_gen:
        raise ValueError(
            f"After alignment raw_target first dim must equal T_gen={t_gen}, got {raw_target.shape[0]}"
        )
    return raw_target


def hellinger(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    d = (1.0 / (2.0**0.5)) * torch.linalg.norm(torch.sqrt(p) - torch.sqrt(q), ord=2)
    return float(d.item())


def weights_from_importance(row: torch.Tensor) -> torch.Tensor:
    row = torch.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
    row = torch.clamp(row, min=0.0)
    m = float(row.max().item()) if row.numel() > 0 else 0.0
    if m <= 0.0:
        return torch.zeros_like(row)
    return torch.clamp(row / m, 0.0, 1.0)


@torch.inference_mode()
def compute_soft_norm_metrics_a4(
    model: Any,
    source_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    importance_map: torch.Tensor,
    metric_stride: int = 1,
    debug_steps: bool = False,
) -> Dict[str, Any]:
    if source_ids.ndim != 1 or generated_ids.ndim != 1:
        raise ValueError("source_ids and generated_ids must be 1D tensors")
    if importance_map.ndim != 2:
        raise ValueError(f"importance_map must be 2D, got {tuple(importance_map.shape)}")

    model.eval()
    device = next(model.parameters()).device
    emb_layer = model.get_input_embeddings()
    emb_dtype = emb_layer.weight.dtype
    # Extra guard for MPS mixed-dtype failures in matmul kernels.
    if device.type == "mps":
        emb_dtype = torch.float32

    source_ids = source_ids.to(torch.long)
    generated_ids = generated_ids.to(torch.long)
    importance_map = torch.nan_to_num(importance_map.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)

    l_in = int(source_ids.shape[0])
    l_total = int(generated_ids.shape[0])
    t_gen = l_total - l_in
    if t_gen <= 0:
        raise ValueError(f"No generated tokens (L_total={l_total}, L_in={l_in})")
    if importance_map.shape[0] != t_gen:
        raise ValueError(f"importance_map first dim mismatch: {importance_map.shape[0]} != T_gen={t_gen}")
    w = int(importance_map.shape[1])
    if w <= 0 or w > l_total:
        raise ValueError(f"importance_map width must be in [1, {l_total}], got {w}")

    e_base = emb_layer.weight.mean(dim=0).to(device=device, dtype=emb_dtype).view(1, 1, -1)

    target_pos_list: List[int] = []
    target_token_id_list: List[int] = []
    soft_ns_list: List[float] = []
    soft_nc_list: List[float] = []
    dp0_list: List[float] = []
    dpr_list: List[float] = []
    dpnotr_list: List[float] = []
    step_debug: List[Dict[str, Any]] = []

    near_zero_notr = 0
    near_equal_r_0 = 0
    steps = 0

    stride = max(1, int(metric_stride))
    for target_pos in range(l_in, l_total, stride):
        step_i = target_pos - l_in
        ctx_ids = generated_ids[:target_pos].unsqueeze(0).to(device)
        l_ctx = int(ctx_ids.shape[1])

        row = importance_map[step_i]
        use_len = min(int(row.shape[0]), l_ctx)
        aligned = torch.zeros((l_ctx,), dtype=torch.float32)
        aligned[:use_len] = row[:use_len].to(torch.float32)

        w_r = weights_from_importance(aligned)
        w_notr = 1.0 - w_r
        w_zero = torch.zeros_like(w_r)
        w_full = torch.ones_like(w_r)

        e_full = emb_layer(ctx_ids).to(dtype=emb_dtype)
        e_full = e_full.expand(4, -1, -1)

        w_batch = torch.stack([w_full, w_r, w_notr, w_zero], dim=0).to(device=device, dtype=emb_dtype).unsqueeze(-1)
        inputs_embeds_batch = w_batch * e_full + (1.0 - w_batch) * e_base
        attention_mask_batch = torch.ones((4, l_ctx), dtype=torch.long, device=device)

        out = model(inputs_embeds=inputs_embeds_batch, attention_mask=attention_mask_batch)
        logits_last = out.logits[:, -1, :].to(torch.float32)
        probs = F.softmax(logits_last, dim=-1)

        p_full, p_r, p_notr, p_0 = probs[0], probs[1], probs[2], probs[3]
        dpr = hellinger(p_full, p_r)
        dpnotr = hellinger(p_full, p_notr)
        dp0 = hellinger(p_full, p_0)

        if dp0 <= 0.0:
            soft_ns = 0.0
            soft_nc = 0.0
        else:
            soft_ns = max(0.0, (dp0 - dpr) / dp0)
            soft_nc = dpnotr / dp0

        if dpnotr < 1e-10:
            near_zero_notr += 1
        if abs(dpr - dp0) < 1e-10:
            near_equal_r_0 += 1
        steps += 1

        target_pos_list.append(int(target_pos))
        target_token_id_list.append(int(generated_ids[target_pos].item()))
        soft_ns_list.append(float(soft_ns))
        soft_nc_list.append(float(soft_nc))
        dp0_list.append(float(dp0))
        dpr_list.append(float(dpr))
        dpnotr_list.append(float(dpnotr))

        if debug_steps:
            step_debug.append(
                {
                    "target_pos": int(target_pos),
                    "l_ctx": l_ctx,
                    "use_len": use_len,
                    "importance_min": float(aligned.min().item()),
                    "importance_max": float(aligned.max().item()),
                    "w_min": float(w_r.min().item()),
                    "w_max": float(w_r.max().item()),
                    "w_mean": float(w_r.mean().item()),
                    "w_nonzero_ratio": float((w_r > 0).float().mean().item()),
                }
            )

    warnings: List[str] = []
    if steps > 0 and (near_zero_notr / steps) > 0.9 and (near_equal_r_0 / steps) > 0.9:
        warnings.append("degenerate_scores_detected: dPnotR~0 and dPR~dP0 for most steps")

    return {
        "target_pos": target_pos_list,
        "target_token_id": target_token_id_list,
        "soft_ns": soft_ns_list,
        "soft_nc": soft_nc_list,
        "dP0": dp0_list,
        "dPR": dpr_list,
        "dPnotR": dpnotr_list,
        "soft_ns_mean": float(sum(soft_ns_list) / max(1, len(soft_ns_list))),
        "soft_nc_mean": float(sum(soft_nc_list) / max(1, len(soft_nc_list))),
        "warnings": warnings,
        "metric_stride": stride,
        "a4_batching": True,
        "debug_steps": step_debug if debug_steps else None,
    }


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


def validate_configs(base_cfg: Dict[str, Any], grid_cfg: Dict[str, Any]) -> None:
    for key in ["models", "datasets", "attribution_functions", "dimensionality_reduction_methods"]:
        if key not in grid_cfg:
            raise ValueError(f"grid config is missing required key: '{key}'")
        if not isinstance(grid_cfg[key], list):
            raise TypeError(f"grid key '{key}' must be a list")
    if "paths" not in base_cfg or "output_dir" not in base_cfg.get("paths", {}):
        raise ValueError("base config must define paths.output_dir")
    for d in grid_cfg["datasets"]:
        p = d.get("path")
        if not p:
            raise ValueError(f"dataset entry missing path: {d}")
        if not Path(p).exists():
            raise FileNotFoundError(f"dataset path does not exist: {p}")
    for a in grid_cfg["attribution_functions"]:
        name = (a.get("name") or "").lower()
        if name not in ALLOWED_ATTR_METHODS:
            raise ValueError(f"Unknown attribution method '{name}'. Allowed: {sorted(ALLOWED_ATTR_METHODS)}")
        if "params" in a and not isinstance(a["params"], dict):
            raise TypeError(f"attribution params must be dict for '{name}'")
    for d in grid_cfg["dimensionality_reduction_methods"]:
        name = (d.get("name") or "").lower()
        if name not in ALLOWED_DIMRED_METHODS:
            raise ValueError(f"Unknown dimred method '{name}'. Allowed: {sorted(ALLOWED_DIMRED_METHODS)}")
        if "params" in d and not isinstance(d["params"], dict):
            raise TypeError(f"dimred params must be dict for '{name}'")


def filter_methods(items: List[Dict[str, Any]], only: Optional[str]) -> List[Dict[str, Any]]:
    if not only:
        return items
    want = {x.strip().lower() for x in only.split(",") if x.strip()}
    return [x for x in items if (x.get("name") or "").lower() in want]


def switch_attr_method_if_supported(inseq_model: Any, method_name: str) -> None:
    if hasattr(inseq_model, "load_attribution_method"):
        inseq_model.load_attribution_method(method_name)
        return
    # No-op if runtime switching is unavailable; caller can provide a model
    # instance that was loaded with the desired attribution method.
    return


def run_attr(
    inseq_model: Any,
    prompt: str,
    full_text: str,
    method_name: str,
    attr_params: Dict[str, Any],
) -> Any:
    switch_attr_method_if_supported(inseq_model, method_name)
    return inseq_model.attribute(
        input_texts=prompt,
        generated_texts=full_text,
        show_progress=False,
        **(attr_params or {}),
    )


def build_error_payload(stage: str, prompt_idx: int, error: Exception, **context: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": stage,
        "prompt_idx": prompt_idx,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    payload.update(context)
    return payload
