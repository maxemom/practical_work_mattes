from __future__ import annotations

from dataclasses import dataclass, field
import gc
from time import perf_counter
from typing import Any, Dict, List

import torch

from pwm.typess import MethodResult, PromptRunResult
from pwm.utils_attribution_V2 import get_raw_targets_lxt_v2, get_raw_targets_v2
from pwm.utils_dimred import reduce_raw_targets_to_importance
from pwm.utils_metrics_strict import compute_strict_soft_metrics
from pwm.utils_pipeline import (
    clear_device_cache,
    configure_runtime_warnings,
    disable_loading_verbosity,
    ensure_pad_token,
    hf_generate_once,
    safe_name,
    set_global_seed,
    register_inseq_model_configs,
    stabilize_model_for_metrics,
    switch_attr_method_if_supported,
)
from pwm.utils_runtime import resolve_model_dtype


def _seed_for(base_seed: int, prompt_idx: int, attr_idx: int = 0, dim_idx: int = 0, stage_offset: int = 0) -> int:
    return (
        int(base_seed)
        + 1_000_000 * int(prompt_idx)
        + 10_000 * int(attr_idx)
        + 100 * int(dim_idx)
        + int(stage_offset)
    )


def _tensor_to_optional_list(tensor: torch.Tensor) -> List[List[float | None]]:
    rows: List[List[float | None]] = []
    cpu = tensor.detach().cpu().to(torch.float32)
    for row in cpu:
        out_row: List[float | None] = []
        for value in row.tolist():
            if value != value:
                out_row.append(None)
            else:
                out_row.append(float(value))
        rows.append(out_row)
    return rows


def _make_combo_key(attr_tag: str, dimred_tag: str) -> str:
    return f"{attr_tag}__{dimred_tag}"


def _make_skipped_method_result(
    attr_cfg: Dict[str, Any],
    dim_cfg: Dict[str, Any],
    reason: str,
) -> MethodResult:
    attr_tag = str(attr_cfg.get("tag") or safe_name(attr_cfg.get("name", "attr")))
    dim_tag = str(dim_cfg.get("tag") or safe_name(dim_cfg.get("name", "dimred")))
    return MethodResult(
        combo_key=_make_combo_key(attr_tag, dim_tag),
        attribution_tag=attr_tag,
        attribution_name=str(attr_cfg.get("name", "")),
        attribution_params=dict(attr_cfg.get("params", {}) or {}),
        dimred_tag=dim_tag,
        dimred_name=str(dim_cfg.get("name", "")),
        dimred_params=dict(dim_cfg.get("params", {}) or {}),
        importance_scores=[],
        soft_ns_per_token=[],
        soft_nc_per_token=[],
        final_sufficiency_per_token=[],
        final_comprehensiveness_per_token=[],
        random_soft_ns_per_token=[],
        random_soft_nc_per_token=[],
        soft_ns_mean=0.0,
        soft_nc_mean=0.0,
        final_sufficiency_mean=0.0,
        final_comprehensiveness_mean=0.0,
        target_pos=[],
        target_token_ids=[],
        target_token_texts=[],
        warnings=[],
        skipped=True,
        skip_reason=reason,
    )


@dataclass
class ExperimentRuntime:
    model_name: str
    device: str
    model_dtype_name: str
    hf_model: Any
    tokenizer: Any
    primary_attr_name: str | None
    primary_inseq_model: Any | None = None
    can_switch_runtime: bool = False
    attr_cache: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, model_name: str, resolved_config: Dict[str, Any], attribution_methods: List[Dict[str, Any]]) -> "ExperimentRuntime":
        device = str(resolved_config.get("runtime", {}).get("device", "cpu"))
        disable_loading_verbosity()
        configure_runtime_warnings()
        register_inseq_model_configs()
        model_dtype, model_dtype_name = resolve_model_dtype(resolved_config)
        non_lxt = [cfg for cfg in attribution_methods if str(cfg.get("name", "")).lower() != "lxt"]

        if non_lxt:
            import inseq

            primary_attr = str(non_lxt[0]["name"])
            load_kwargs: Dict[str, Any] = {}
            if model_dtype is not None:
                load_kwargs["model_kwargs"] = {"torch_dtype": model_dtype}
            try:
                inseq_model = inseq.load_model(
                    model_name,
                    attribution_method=primary_attr,
                    device=device,
                    **load_kwargs,
                )
            except TypeError:
                inseq_model = inseq.load_model(
                    model_name,
                    attribution_method=primary_attr,
                    device=device,
                )
            tokenizer = inseq_model.tokenizer
            hf_model = inseq_model.model
            if device == "mps":
                hf_model = stabilize_model_for_metrics(hf_model)
                inseq_model.model = hf_model
            ensure_pad_token(tokenizer)
            return cls(
                model_name=model_name,
                device=device,
                model_dtype_name=model_dtype_name,
                hf_model=hf_model,
                tokenizer=tokenizer,
                primary_attr_name=primary_attr,
                primary_inseq_model=inseq_model,
                can_switch_runtime=hasattr(inseq_model, "load_attribution_method"),
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ensure_pad_token(tokenizer)
        model_kwargs: Dict[str, Any] = {}
        if model_dtype is not None:
            model_kwargs["torch_dtype"] = model_dtype
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        hf_model.to(device)
        if device == "mps":
            hf_model = stabilize_model_for_metrics(hf_model)
        hf_model.eval()
        return cls(
            model_name=model_name,
            device=device,
            model_dtype_name=model_dtype_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            primary_attr_name=None,
            primary_inseq_model=None,
            can_switch_runtime=False,
        )

    def get_inseq_model(self, attr_name: str) -> Any:
        if attr_name.lower() == "lxt":
            raise ValueError("LXT does not use an inseq runtime.")

        if self.primary_inseq_model is not None:
            if self.can_switch_runtime:
                switch_attr_method_if_supported(self.primary_inseq_model, attr_name)
                return self.primary_inseq_model
            if attr_name == self.primary_attr_name:
                return self.primary_inseq_model

        if attr_name not in self.attr_cache:
            import inseq

            load_kwargs: Dict[str, Any] = {}
            resolved_dtype = None
            if self.model_dtype_name in {"float16", "bfloat16", "float32"}:
                resolved_dtype = getattr(torch, self.model_dtype_name)
            if resolved_dtype is not None:
                load_kwargs["model_kwargs"] = {"torch_dtype": resolved_dtype}
            try:
                self.attr_cache[attr_name] = inseq.load_model(
                    self.model_name,
                    attribution_method=attr_name,
                    device=self.device,
                    **load_kwargs,
                )
            except TypeError:
                self.attr_cache[attr_name] = inseq.load_model(
                    self.model_name,
                    attribution_method=attr_name,
                    device=self.device,
                )
            if self.device == "mps":
                self.attr_cache[attr_name].model = stabilize_model_for_metrics(self.attr_cache[attr_name].model)
        return self.attr_cache[attr_name]

    def move_to_device(self, target_device: str) -> None:
        normalized_target = str(target_device or "").strip().lower()
        if not normalized_target or normalized_target == self.device:
            return

        if normalized_target == "mps":
            mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if not mps_ok:
                raise RuntimeError("Requested device 'mps' for LXT, but MPS is not available.")
        if normalized_target.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{normalized_target}' for LXT, but CUDA is not available.")

        self.hf_model.to(normalized_target)
        if normalized_target == "mps":
            self.hf_model = stabilize_model_for_metrics(self.hf_model)
        self.hf_model.eval()
        if self.primary_inseq_model is not None:
            self.primary_inseq_model.model = self.hf_model
            if hasattr(self.primary_inseq_model, "device"):
                self.primary_inseq_model.device = normalized_target

        for inseq_model in self.attr_cache.values():
            if hasattr(inseq_model, "model"):
                inseq_model.model.to(normalized_target)
                if normalized_target == "mps":
                    inseq_model.model = stabilize_model_for_metrics(inseq_model.model)
                inseq_model.model.eval()
            if hasattr(inseq_model, "device"):
                inseq_model.device = normalized_target

        self.device = normalized_target

    def close(self) -> None:
        for model in self.attr_cache.values():
            del model
        self.attr_cache.clear()
        if self.primary_inseq_model is not None:
            del self.primary_inseq_model
            self.primary_inseq_model = None
        gc.collect()
        clear_device_cache(self.device)


def run_prompt_experiment(
    prompt: str,
    model_name: str,
    resolved_config: Dict[str, Any],
    attribution_methods: List[Dict[str, Any]],
    dimred_methods: List[Dict[str, Any]],
    *,
    prompt_idx: int,
    dataset_name: str,
    runtime: ExperimentRuntime,
) -> PromptRunResult:
    base_seed = int(resolved_config.get("seeds", {}).get("seed", 42))
    generation_seed = _seed_for(base_seed, prompt_idx, stage_offset=1)
    set_global_seed(generation_seed)

    source_ids, total_ids, generated_text = hf_generate_once(
        model=runtime.hf_model,
        tokenizer=runtime.tokenizer,
        prompt=prompt,
        generation_cfg=resolved_config.get("generation", {}),
    )

    source_len = int(source_ids.shape[0])
    total_len = int(total_ids.shape[0])
    generated_token_ids = total_ids[source_len:].tolist()
    generated_tokens = [
        runtime.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        for token_id in generated_token_ids
    ]

    result = PromptRunResult(
        prompt_idx=prompt_idx,
        prompt=prompt,
        model_name=model_name,
        dataset_name=dataset_name,
        generated_text=generated_text,
        source_ids=source_ids.tolist(),
        total_ids=total_ids.tolist(),
        generated_token_ids=generated_token_ids,
        source_len=source_len,
        total_len=total_len,
        generated_tokens=generated_tokens,
    )

    if total_len <= source_len:
        result.warnings.append("no_generated_tokens")
        for attr_cfg in attribution_methods:
            for dim_cfg in dimred_methods:
                skipped = _make_skipped_method_result(attr_cfg, dim_cfg, "no_generated_tokens")
                result.combinations[skipped.combo_key] = skipped
        return result

    total_ids_cpu = total_ids.detach().cpu().to(torch.long)

    for attr_idx, attr_cfg in enumerate(attribution_methods):
        attr_name = str(attr_cfg.get("name", ""))
        attr_params = dict(attr_cfg.get("params", {}) or {})
        attr_tag = str(attr_cfg.get("tag") or safe_name(attr_name))
        attr_device = runtime.device
        attr_elapsed_ms: float | None = None
        attr_step_times_ms: List[float] = []

        try:
            set_global_seed(_seed_for(base_seed, prompt_idx, attr_idx=attr_idx, stage_offset=11))
            attr_started = perf_counter()
            if attr_name.lower() == "lxt":
                requested_lxt_device = str(attr_params.get("device", "")).strip().lower()
                if requested_lxt_device:
                    runtime.move_to_device(requested_lxt_device)
                attr_device = runtime.device
                raw_attr_result = get_raw_targets_lxt_v2(
                    model=runtime.hf_model,
                    generated_ids=total_ids_cpu,
                    source_len=source_len,
                    attr_params=attr_params,
                )
                raw_attr = raw_attr_result.raw_target
                attr_device = raw_attr_result.device or runtime.device
                attr_elapsed_ms = raw_attr_result.elapsed_ms
                attr_step_times_ms = list(raw_attr_result.step_times_ms or [])
                if attr_elapsed_ms is None:
                    attr_elapsed_ms = (perf_counter() - attr_started) * 1000.0
                print(
                    f"[attr] method={attr_name} prompt={prompt_idx} elapsed_ms={attr_elapsed_ms:.2f} "
                    f"steps={len(attr_step_times_ms)} device={attr_device}",
                    flush=True,
                )
            else:
                inseq_model = runtime.get_inseq_model(attr_name)
                raw_attr_result = get_raw_targets_v2(
                    inseq_model=inseq_model,
                    prompt=prompt,
                    generated_ids=total_ids_cpu,
                    source_len=source_len,
                    attr_name=attr_name,
                    attr_params=attr_params,
                )
                raw_attr = raw_attr_result.raw_target
                attr_device = runtime.device
                attr_elapsed_ms = (perf_counter() - attr_started) * 1000.0
                print(
                    f"[attr] method={attr_name} prompt={prompt_idx} elapsed_ms={attr_elapsed_ms:.2f} "
                    f"device={attr_device}",
                    flush=True,
                )
        except Exception as exc:
            reason = f"attribution_failed:{exc}"
            for dim_cfg in dimred_methods:
                skipped = _make_skipped_method_result(
                    {**attr_cfg, "tag": attr_tag},
                    dim_cfg,
                    reason,
                )
                result.combinations[skipped.combo_key] = skipped
            continue

        for dim_idx, dim_cfg in enumerate(dimred_methods):
            dim_name = str(dim_cfg.get("name", ""))
            dim_params = dict(dim_cfg.get("params", {}) or {})
            dim_tag = str(dim_cfg.get("tag") or safe_name(dim_name))
            combo_key = _make_combo_key(attr_tag, dim_tag)
            dimred_elapsed_ms: float | None = None
            metrics_elapsed_ms: float | None = None
            combo_elapsed_ms: float | None = None

            try:
                combo_started = perf_counter()
                dimred_started = perf_counter()
                importance_map = reduce_raw_targets_to_importance(
                    raw_attr,
                    dim_name,
                    dim_params,
                    seed=_seed_for(base_seed, prompt_idx, attr_idx=attr_idx, dim_idx=dim_idx, stage_offset=21),
                )
                dimred_elapsed_ms = (perf_counter() - dimred_started) * 1000.0

                metrics_started = perf_counter()
                metric_result = compute_strict_soft_metrics(
                    runtime.hf_model,
                    total_ids_cpu,
                    source_len,
                    importance_map,
                    seed=_seed_for(base_seed, prompt_idx, attr_idx=attr_idx, dim_idx=dim_idx, stage_offset=31),
                    eps=float(resolved_config.get("metrics", {}).get("eps", 1e-12)),
                )
                metrics_elapsed_ms = (perf_counter() - metrics_started) * 1000.0
                combo_elapsed_ms = (perf_counter() - combo_started) * 1000.0
                print(
                    f"[combo] method={attr_name} dimred={dim_name} prompt={prompt_idx} "
                    f"attr_ms={float(attr_elapsed_ms or 0.0):.2f} dimred_ms={dimred_elapsed_ms:.2f} "
                    f"metrics_ms={metrics_elapsed_ms:.2f} combo_ms={combo_elapsed_ms:.2f}",
                    flush=True,
                )
                result.combinations[combo_key] = MethodResult(
                    combo_key=combo_key,
                    attribution_tag=attr_tag,
                    attribution_name=attr_name,
                    attribution_params=attr_params,
                    dimred_tag=dim_tag,
                    dimred_name=dim_name,
                    dimred_params=dim_params,
                    importance_scores=_tensor_to_optional_list(importance_map),
                    soft_ns_per_token=metric_result.soft_ns_per_token,
                    soft_nc_per_token=metric_result.soft_nc_per_token,
                    final_sufficiency_per_token=metric_result.final_sufficiency_per_token,
                    final_comprehensiveness_per_token=metric_result.final_comprehensiveness_per_token,
                    random_soft_ns_per_token=metric_result.random_soft_ns_per_token,
                    random_soft_nc_per_token=metric_result.random_soft_nc_per_token,
                    soft_ns_mean=metric_result.soft_ns_mean,
                    soft_nc_mean=metric_result.soft_nc_mean,
                    final_sufficiency_mean=metric_result.final_sufficiency_mean,
                    final_comprehensiveness_mean=metric_result.final_comprehensiveness_mean,
                    target_pos=metric_result.target_pos,
                    target_token_ids=metric_result.target_token_id,
                    target_token_texts=[
                        runtime.tokenizer.decode([int(token_id)], skip_special_tokens=False)
                        for token_id in metric_result.target_token_id
                    ],
                    warnings=metric_result.warnings,
                    attribution_device=attr_device,
                    attribution_elapsed_ms=attr_elapsed_ms,
                    attribution_step_times_ms=list(attr_step_times_ms),
                    dimred_elapsed_ms=dimred_elapsed_ms,
                    metrics_elapsed_ms=metrics_elapsed_ms,
                    combo_elapsed_ms=combo_elapsed_ms,
                )
            except Exception as exc:
                result.combinations[combo_key] = _make_skipped_method_result(
                    {**attr_cfg, "tag": attr_tag},
                    {**dim_cfg, "tag": dim_tag},
                    f"dimred_or_metric_failed:{exc}",
                )

        del raw_attr
        gc.collect()
        clear_device_cache(runtime.device)

    return result
