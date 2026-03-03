from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Ensure local package import works without editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pwm.utils_aggregation import aggregate_baseline
from pwm.utils_base import deep_merge, load_yaml
from pwm.utils_dimred import aggregate_dimred
from pwm.utils_pipeline import (
    append_error,
    build_attr_index,
    build_dimred_index,
    build_error_payload,
    clear_device_cache,
    compute_soft_norm_metrics_a4,
    extract_raw_target_with_alignment,
    filter_methods,
    hf_generate_once,
    importance_stats,
    load_prompts,
    raw_target_nan_stats,
    resolve_device,
    run_attr,
    safe_name,
    set_global_seed,
    stabilize_model_for_metrics,
    validate_configs,
)
from pwm.utils_results import save_json, save_softnorm_steps_csv


def _version_or_na(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except Exception:
        return "n/a"


def _configure_loading_verbosity(show_loading: bool) -> None:
    if show_loading:
        return

    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass


def _ts_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_stage_start(prompt_idx: int, stage: str, extra: str = "") -> float:
    msg = f"[time] {_ts_now()} | prompt_{prompt_idx:03d} | {stage} START"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)
    return time.perf_counter()


def _log_stage_end(prompt_idx: int, stage: str, t0: float, extra: str = "") -> None:
    elapsed_s = time.perf_counter() - t0
    msg = f"[time] {_ts_now()} | prompt_{prompt_idx:03d} | {stage} END | elapsed_s={elapsed_s:.3f}"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)


def _is_mps_dtype_mm_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        "MPSNDArrayMatrixMultiplication" in msg
        or "cannot have different datatype" in msg
        or "Destination NDArray and Accumulator NDArray cannot have different datatype" in msg
    )


def _needs_mps_fp32_for_attr(attr_name: str) -> bool:
    # Methods using captum internals / sampling are more likely to trigger
    # mixed-dtype matmul assertions on MPS.
    return (attr_name or "").lower() in {
        "gradient_shap",
        "deeplift",
        "integrated_gradients",
    }


def _prepare_attr_params_for_device(
    attr_name: str,
    attr_params: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    params = dict(attr_params or {})
    name = (attr_name or "").lower()
    if device == "mps" and name == "gradient_shap":
        n_samples = int(params.get("n_samples", 24))
        if n_samples > 8:
            params["n_samples"] = 8
            print(
                f"[warn] gradient_shap on MPS: reducing n_samples {n_samples} -> 8 to avoid OOM/system slowdown.",
                flush=True,
            )
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="configs/base.yaml")
    parser.add_argument("--grid", type=str, default="configs/grid.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--only-attr", type=str, default=None)
    parser.add_argument("--only-dimred", type=str, default=None)
    parser.add_argument("--only-prompt-idx", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional runtime device override (e.g., auto|cpu|mps|cuda|cuda:0).",
    )
    parser.add_argument(
        "--show-loading",
        action="store_true",
        help="Show model/tokenizer loading progress bars and logs.",
    )
    args = parser.parse_args()

    _configure_loading_verbosity(show_loading=args.show_loading)
    warnings.filterwarnings(
        "ignore",
        message="Setting forward, backward hooks and attributes on non-linear",
        category=UserWarning,
    )

    base_cfg = load_yaml(Path(args.base))
    grid_cfg = load_yaml(Path(args.grid))
    validate_configs(base_cfg, grid_cfg)

    attrs = filter_methods(grid_cfg["attribution_functions"], args.only_attr)
    dimreds = filter_methods(grid_cfg["dimensionality_reduction_methods"], args.only_dimred)
    if not attrs:
        raise ValueError("No attribution methods left after filtering (--only-attr).")
    if not dimreds:
        raise ValueError("No dimred methods left after filtering (--only-dimred).")

    output_root = Path(base_cfg["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    base_seed = int(base_cfg.get("seeds", {}).get("seed", 42))
    metric_stride = int(base_cfg.get("metrics", {}).get("metric_stride", 1))
    debug_metric_steps = bool(base_cfg.get("metrics", {}).get("debug_steps", False))

    requested_device = args.device if args.device is not None else str(base_cfg.get("runtime", {}).get("device", "auto"))
    chosen_device = resolve_device(requested_device)
    base_cfg.setdefault("runtime", {})
    base_cfg["runtime"]["device"] = chosen_device

    print("=== Resolved Startup Config ===")
    print(f"base_seed={base_seed} | metric_stride={metric_stride} | device={chosen_device}")
    print(f"attribution_methods={[a.get('name') for a in attrs]}")
    print(f"dimred_methods={[d.get('name') for d in dimreds]}")

    models = grid_cfg["models"]
    datasets = grid_cfg["datasets"]
    total_runs = len(models) * len(datasets)
    print(f"Planned outer runs (model,dataset): {total_runs}")

    if args.dry_run:
        return

    import inseq
    from transformers import AutoConfig

    for m in models:
        mn = m.get("name")
        try:
            AutoConfig.from_pretrained(mn)
        except Exception as e:
            raise RuntimeError(f"Model config not loadable for '{mn}': {e}") from e
    for d in datasets:
        ds_prompts = load_prompts(d, args.max_prompts)
        if not ds_prompts:
            raise RuntimeError(f"Dataset '{d.get('name', 'unknown')}' produced zero prompts")

    run_i = 0
    for model_cfg in models:
        for dataset_cfg in datasets:
            run_i += 1
            model_name = model_cfg["name"]
            dataset_name = dataset_cfg["name"]
            model_slug = safe_name(model_name)
            dataset_slug = safe_name(dataset_name)
            run_dir = output_root / model_slug / dataset_slug
            run_dir.mkdir(parents=True, exist_ok=True)
            prompts_root = run_dir / "prompts"
            prompts_root.mkdir(parents=True, exist_ok=True)

            resolved = deep_merge(
                base_cfg,
                {
                    "model": model_cfg,
                    "dataset": dataset_cfg,
                    "attribution_functions": attrs,
                    "dimensionality_reduction_methods": dimreds,
                },
            )
            with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
                yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)

            attr_index = build_attr_index(attrs)
            dimred_index = build_dimred_index(dimreds)
            save_json(run_dir / "attr_index.json", attr_index)
            save_json(run_dir / "dimred_index.json", dimred_index)

            run_meta = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version.split()[0],
                "torch": _version_or_na("torch"),
                "transformers": _version_or_na("transformers"),
                "inseq": _version_or_na("inseq"),
                "requested_device": requested_device,
                "device": chosen_device,
                "model_name": model_name,
                "dataset_name": dataset_name,
            }
            save_json(run_dir / "run_meta.json", run_meta)

            print(f"[{run_i}/{total_runs}] model={model_name} dataset={dataset_name}")
            prompts = load_prompts(dataset_cfg, args.max_prompts)
            if args.only_prompt_idx is not None:
                if args.only_prompt_idx < 0 or args.only_prompt_idx >= len(prompts):
                    raise IndexError(f"--only-prompt-idx {args.only_prompt_idx} out of range [0,{len(prompts)-1}]")
                prompts = [prompts[args.only_prompt_idx]]
                prompt_start_idx = args.only_prompt_idx
            else:
                prompt_start_idx = 0

            first_attr_name = attrs[0]["name"]
            inseq_model = inseq.load_model(
                model_name,
                attribution_method=first_attr_name,
                device=chosen_device,
            )
            hf_model = inseq_model.model
            tokenizer = inseq_model.tokenizer

            # Compat: if runtime switching is unavailable, avoid preloading all
            # attribution models at once (high memory pressure on MPS).
            can_switch_runtime = hasattr(inseq_model, "load_attribution_method")
            if not can_switch_runtime and len(attrs) > 1:
                print("[warn] inseq model has no load_attribution_method; loading one attribution model at a time.")

            for local_i, prompt in enumerate(prompts):
                prompt_idx = prompt_start_idx + local_i
                prompt_dir = prompts_root / f"prompt_{prompt_idx:03d}"
                prompt_dir.mkdir(parents=True, exist_ok=True)

                debug_payload: Dict[str, Any] = {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "attr_tags": list(attr_index.keys()),
                    "dimred_tags": list(dimred_index.keys()),
                }
                save_json(prompt_dir / "debug.json", debug_payload)

                try:
                    t_gen_stage = _log_stage_start(prompt_idx, "generation")
                    seed_gen = base_seed + 10_000 * prompt_idx
                    set_global_seed(seed_gen)
                    src_ids, gen_ids, full_text = hf_generate_once(
                        model=hf_model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        generation_cfg=resolved.get("generation", {}),
                    )
                    source_len = int(src_ids.shape[0])
                    full_len = int(gen_ids.shape[0])
                    t_gen = full_len - source_len
                    if t_gen <= 0:
                        print(f"[warn] prompt_{prompt_idx:03d}: no generation (L_total={full_len}, L_in={source_len}), skip.")
                        debug_payload.update(
                            {
                                "seed_gen": seed_gen,
                                "source_len": source_len,
                                "full_len": full_len,
                                "T_gen": t_gen,
                                "source_text": prompt,
                                "full_text": full_text,
                                "generated_ids": gen_ids.tolist(),
                                "skipped_reason": "no_generated_tokens",
                            }
                        )
                        save_json(prompt_dir / "debug.json", debug_payload)
                        _log_stage_end(prompt_idx, "generation", t_gen_stage, "skipped:no_generated_tokens")
                        continue

                    debug_payload.update(
                        {
                            "seed_gen": seed_gen,
                            "source_len": source_len,
                            "full_len": full_len,
                            "T_gen": t_gen,
                            "source_text": prompt,
                            "full_text": full_text,
                            "generated_ids": gen_ids.tolist(),
                            "source_ids": src_ids.tolist(),
                            "attr_debug": {},
                        }
                    )
                    save_json(prompt_dir / "debug.json", debug_payload)
                    _log_stage_end(
                        prompt_idx,
                        "generation",
                        t_gen_stage,
                        f"source_len={source_len} full_len={full_len} T_gen={t_gen}",
                    )
                except Exception as e:
                    _log_stage_end(prompt_idx, "generation", t_gen_stage, "failed")
                    append_error(prompt_dir, build_error_payload("generation", prompt_idx, e))
                    continue

                for a_tag, a_cfg in attr_index.items():
                    t_attr_stage = None
                    try:
                        attr_name = a_cfg["name"]
                        t_attr_stage = _log_stage_start(prompt_idx, "attribution", f"attr_tag={a_tag} method={attr_name}")
                        attr_params = _prepare_attr_params_for_device(attr_name, a_cfg["params"], chosen_device)
                        seed_attr = base_seed + 100_000 * prompt_idx + 1_000 * int(a_cfg["index"])
                        set_global_seed(seed_attr)

                        if can_switch_runtime or attr_name == first_attr_name:
                            model_for_attr = inseq_model
                            temp_model_loaded = False
                        else:
                            model_for_attr = inseq.load_model(
                                model_name,
                                attribution_method=attr_name,
                                device=chosen_device,
                            )
                            temp_model_loaded = True

                        # Hard MPS safety guard for methods that can crash at native level
                        # before Python can catch exceptions.
                        if chosen_device == "mps" and _needs_mps_fp32_for_attr(attr_name):
                            model_for_attr.model = stabilize_model_for_metrics(model_for_attr.model)

                        try:
                            out = run_attr(
                                inseq_model=model_for_attr,
                                prompt=prompt,
                                full_text=full_text,
                                method_name=attr_name,
                                attr_params=attr_params,
                            )
                        except Exception as e:
                            if chosen_device == "mps" and _is_mps_dtype_mm_error(e):
                                print(
                                    f"[warn] prompt_{prompt_idx:03d} {a_tag}: MPS dtype mismatch during attribution; retrying in float32.",
                                    flush=True,
                                )
                                model_for_attr.model = stabilize_model_for_metrics(model_for_attr.model)
                                out = run_attr(
                                    inseq_model=model_for_attr,
                                    prompt=prompt,
                                    full_text=full_text,
                                    method_name=attr_name,
                                    attr_params=attr_params,
                                )
                            else:
                                raise
                        raw_target = extract_raw_target_with_alignment(
                            out=out,
                            source_ids=src_ids,
                            generated_ids=gen_ids,
                        )
                        del out
                        if temp_model_loaded:
                            del model_for_attr
                            gc.collect()
                            clear_device_cache(chosen_device)
                        _log_stage_end(prompt_idx, "attribution", t_attr_stage, f"attr_tag={a_tag}")
                        nan_stats = raw_target_nan_stats(raw_target, source_len=source_len)
                        nan_ratio = nan_stats["global_nan_ratio"]
                        active_nan_ratio = nan_stats["active_nan_ratio"]
                        if active_nan_ratio > 0.05:
                            print(
                                f"[warn] prompt_{prompt_idx:03d} {a_tag}: active_nan_ratio={active_nan_ratio:.4f} > 0.05 "
                                f"(global_nan_ratio={nan_ratio:.4f})"
                            )

                        baseline_target = aggregate_baseline(raw_target=raw_target, resolved=resolved)
                        baseline_stats = importance_stats(baseline_target)
                        if prompt_idx == prompt_start_idx:
                            print(f"[stats] prompt_{prompt_idx:03d} {a_tag} baseline={baseline_stats}")

                        hf_model = stabilize_model_for_metrics(hf_model)
                        t_base_metrics = _log_stage_start(prompt_idx, "metrics_baseline", f"attr_tag={a_tag}")
                        baseline_res = compute_soft_norm_metrics_a4(
                            model=hf_model,
                            source_ids=src_ids,
                            generated_ids=gen_ids,
                            importance_map=baseline_target,
                            metric_stride=metric_stride,
                            debug_steps=debug_metric_steps,
                        )
                        save_json(prompt_dir / f"{a_tag}_baseline.json", baseline_res)
                        save_softnorm_steps_csv(prompt_dir / f"{a_tag}_baseline_steps.csv", baseline_res)
                        _log_stage_end(prompt_idx, "metrics_baseline", t_base_metrics, f"attr_tag={a_tag}")

                        debug_payload["attr_debug"][a_tag] = {
                            "seed_attr": seed_attr,
                            "raw_target_shape": list(raw_target.shape),
                            "nan_ratio": nan_ratio,
                            "active_nan_ratio": active_nan_ratio,
                            "baseline_stats": baseline_stats,
                            "baseline_warnings": baseline_res.get("warnings", []),
                        }
                        save_json(prompt_dir / "debug.json", debug_payload)

                        for d_tag, d_cfg in dimred_index.items():
                            try:
                                t_dim_metrics = _log_stage_start(
                                    prompt_idx,
                                    "metrics_dimred",
                                    f"attr_tag={a_tag} dimred_tag={d_tag}",
                                )
                                dim_cfg = {"name": d_cfg["name"], "params": d_cfg["params"]}
                                dim_target = aggregate_dimred(raw_target=raw_target, resolved=resolved, dimred_cfg=dim_cfg)
                                dim_target = torch.nan_to_num(dim_target, nan=0.0, posinf=0.0, neginf=0.0)
                                dim_stats = importance_stats(dim_target)
                                if prompt_idx == prompt_start_idx:
                                    print(f"[stats] prompt_{prompt_idx:03d} {a_tag}/{d_tag} dimred={dim_stats}")

                                dim_res = compute_soft_norm_metrics_a4(
                                    model=hf_model,
                                    source_ids=src_ids,
                                    generated_ids=gen_ids,
                                    importance_map=dim_target,
                                    metric_stride=metric_stride,
                                    debug_steps=debug_metric_steps,
                                )
                                save_json(prompt_dir / f"{a_tag}_dimred_{d_tag}.json", dim_res)
                                save_softnorm_steps_csv(prompt_dir / f"{a_tag}_dimred_{d_tag}_steps.csv", dim_res)
                                _log_stage_end(
                                    prompt_idx,
                                    "metrics_dimred",
                                    t_dim_metrics,
                                    f"attr_tag={a_tag} dimred_tag={d_tag}",
                                )

                                debug_payload["attr_debug"][a_tag].setdefault("dimred", {})[d_tag] = {
                                    "stats": dim_stats,
                                    "warnings": dim_res.get("warnings", []),
                                }
                                save_json(prompt_dir / "debug.json", debug_payload)
                            except Exception as e:
                                _log_stage_end(
                                    prompt_idx,
                                    "metrics_dimred",
                                    t_dim_metrics,
                                    f"attr_tag={a_tag} dimred_tag={d_tag} failed",
                                )
                                append_error(
                                    prompt_dir,
                                    build_error_payload(
                                        "dimred_metrics",
                                        prompt_idx,
                                        e,
                                        attr_tag=a_tag,
                                        dimred_tag=d_tag,
                                    ),
                                )
                            continue
                    except Exception as e:
                        if t_attr_stage is not None:
                            _log_stage_end(prompt_idx, "attribution", t_attr_stage, f"attr_tag={a_tag} failed")
                        append_error(
                            prompt_dir,
                            build_error_payload("attribution", prompt_idx, e, attr_tag=a_tag),
                        )
                        continue
                    finally:
                        gc.collect()
                        clear_device_cache(chosen_device)
                        print(
                            f"[cache] {_ts_now()} | prompt_{prompt_idx:03d} | cleared after attr_tag={a_tag}",
                            flush=True,
                        )

                gc.collect()
                clear_device_cache(chosen_device)


if __name__ == "__main__":
    main()
