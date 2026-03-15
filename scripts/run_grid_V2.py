from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml
import inseq

# Ensure local package import works without editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pwm.utils_base import load_yaml
from pwm.utils_attribution_V2 import get_raw_targets_v2
from pwm.utils_generation_V2 import generate_output_text, load_generation_components
from pwm.utils_dimred_V2 import reduce_raw_target
from pwm.utils_metrics_V3 import compute_soft_norm_metrics_v3
from pwm.utils_pipeline import (
    build_attr_index,
    build_dimred_index,
    clear_device_cache,
    load_prompts,
    raw_target_nan_stats,
    resolve_device,
    safe_name,
    set_global_seed,
    stabilize_model_for_metrics,
    validate_configs,
)
from pwm.utils_results_V2 import (
    save_baseline_result_v2,
    save_dimred_result_v2,
    save_json_v2,
)
from pwm.utils_runtime import build_resolved_run_config


def _assert_1d_ids(name: str, x: torch.Tensor) -> None:
    """Sanity-Check: IDs muessen 1D und integer sein."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={tuple(x.shape)}")
    if x.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be int tensor, got dtype={x.dtype}")


def _assert_raw_target_shape(
    raw_target: torch.Tensor,
    source_len: int,
    generated_ids: torch.Tensor,
) -> None:
    """
    Sanity-Check fuer neuen Standard:
    raw_target shape == (L_total, T_gen, D)

    Erwartung:
    - L_in = source_len
    - L_total = len(generated_ids)
    - T_gen = L_total - L_in
    """

    if raw_target.ndim != 3:
        raise ValueError(f"raw_target must be 3D, got shape={tuple(raw_target.shape)}")

    l_in = int(source_len)
    l_total = int(generated_ids.shape[0])
    t_gen = l_total - l_in

    if t_gen <= 0:
        raise ValueError(f"Expected generated tokens, got L_total={l_total}, L_in={l_in}")

    if raw_target.shape[0] != l_total:
        raise ValueError(
            f"raw_target dim0 mismatch: got {raw_target.shape[0]}, expected L_total={l_total}"
        )

    if raw_target.shape[1] != t_gen:
        raise ValueError(
            f"raw_target dim1 mismatch: got {raw_target.shape[1]}, expected T_gen={t_gen}"
        )

    if raw_target.shape[2] <= 0:
        raise ValueError(
            f"raw_target dim2 (embed dim) must be >0, got {raw_target.shape[2]}"
        )


def _assert_importance_shape(
    importance_map: torch.Tensor,
    source_len: int,
    generated_ids: torch.Tensor,
) -> None:
    """
    Erwarteter Output fuer Baseline/DimRed:
    importance_map shape == (L_total, L_gen)
    """
    if importance_map.ndim != 2:
        raise ValueError(f"importance_map must be 2D, got shape={tuple(importance_map.shape)}")

    l_in = int(source_len)
    l_total = int(generated_ids.shape[0])
    t_gen = l_total - l_in

    if importance_map.shape[1] != t_gen:
        raise ValueError(
            f"importance_map dim1 mismatch: got {importance_map.shape[1]}, expected T_gen={t_gen}"
        )
    if importance_map.shape[0] != l_total:
        raise ValueError(
            f"importance_map dim0 mismatch: got {importance_map.shape[0]}, expected L_total={l_total}"
        )


def _build_combinations(models: List[Dict[str, Any]], datasets: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Erzeuge explizite (model_idx, dataset_idx)-Kombinationen."""
    combos: List[Tuple[int, int]] = []
    for mi in range(len(models)):
        for di in range(len(datasets)):
            combos.append((mi, di))
    return combos


def _generation_seed(base_seed: int, prompt_idx: int) -> int:
    return base_seed + 10_000 * prompt_idx


def _attribution_seed(base_seed: int, prompt_idx: int, attr_idx: int) -> int:
    return base_seed + 100_000 * prompt_idx + 1_000 * attr_idx


def _baseline_metrics_seed(base_seed: int, prompt_idx: int, attr_idx: int) -> int:
    return base_seed + 1_000_000 * prompt_idx + 10_000 * attr_idx + 1


def _dimred_metrics_seed(base_seed: int, prompt_idx: int, attr_idx: int, dimred_idx: int) -> int:
    return base_seed + 1_000_000 * prompt_idx + 10_000 * attr_idx + 100 * dimred_idx + 2


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


def _needs_mps_fp32_for_attr(attr_name: str) -> bool:
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
    if device == "mps" and (attr_name or "").lower() == "gradient_shap":
        n_samples = int(params.get("n_samples", 24))
        if n_samples > 8:
            params["n_samples"] = 8
            print(
                f"[warn] gradient_shap on MPS: reducing n_samples {n_samples} -> 8 to avoid unstable execution.",
                flush=True,
            )
    return params


def _is_known_value_zeroing_incompatibility(attr_name: str, exc: Exception) -> bool:
    if (attr_name or "").lower() != "value_zeroing":
        return False
    msg = str(exc)
    return (
        "unsupported operand type(s) for +: 'Tensor' and 'tuple'" in msg
        or "hook" in msg.lower()
    )

def l2_norm_over_dim(raw_target: torch.Tensor) -> torch.Tensor:
    """
    Input:
        raw_target: (L_total, T_gen, D)

    Output:
        l2_map: (L_total, T_gen)

    Berechnet die L2-Norm über D.
    NaNs werden NICHT verändert und propagieren automatisch.
    """

    if raw_target.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {raw_target.shape}")

    l2_map = torch.norm(raw_target, p=2, dim=-1)

    return l2_map

def column_softmax_with_nan(x: torch.Tensor) -> torch.Tensor:
    """
    Input:
        x: (L_total, T_gen)

    Output:
        softmax_map: (L_total, T_gen)

    Softmax wird pro Spalte (T_gen) berechnet.
    NaN-Werte werden als -inf behandelt → Softmax = 0.
    """

    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {x.shape}")

    # NaN -> -inf
    x_safe = torch.where(torch.isnan(x), torch.tensor(float("-inf"), device=x.device), x)

    # numerisch stabile Softmax
    max_per_col = torch.max(x_safe, dim=0, keepdim=True).values
    exp = torch.exp(x_safe - max_per_col)

    softmax = exp / torch.sum(exp, dim=0, keepdim=True)

    return softmax


def main() -> None:
    parser = argparse.ArgumentParser(description="Step-by-step Scaffold fuer die neue Attribution-Pipeline.")
    parser.add_argument("--base", type=str, default="configs/base.yaml")
    parser.add_argument("--grid", type=str, default="configs/grid.yaml")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--show-loading",
        action="store_true",
        help="Show model/tokenizer loading logs and progress bars.",
    )
    args = parser.parse_args()

    _configure_loading_verbosity(show_loading=args.show_loading)

    base_cfg = load_yaml(Path(args.base))
    grid_cfg = load_yaml(Path(args.grid))
    # validate_configs(base_cfg, grid_cfg)

    output_root = Path(base_cfg["paths"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    requested_device = args.device if args.device is not None else str(base_cfg.get("runtime", {}).get("device", "auto"))
    chosen_device = resolve_device(requested_device)

    models = grid_cfg["models"]
    datasets = grid_cfg["datasets"]
    attrs = grid_cfg["attribution_functions"]
    dimreds = grid_cfg["dimensionality_reduction_methods"]

    attr_index = build_attr_index(attrs)
    dimred_index = build_dimred_index(dimreds)
    combos = _build_combinations(models, datasets)
    base_seed = int(base_cfg.get("seeds", {}).get("seed", 42))

    print("=== run_grid_V2 scaffold ===")
    print(f"device={chosen_device} | combinations={len(combos)} | base_seed={base_seed}")

    if args.dry_run:
        return

    # OUTER LOOP: Kombinationen aus Model x Dataset.
    # Erwarteter Output dieser Schleife:
    # - Pro Kombination ein Run-Ordner mit resolved config + indices.
    for run_i, (mi, di) in enumerate(combos, start=1):
        model_cfg = models[mi]
        dataset_cfg = datasets[di]

        model_name = model_cfg["name"]
        dataset_name = dataset_cfg["name"]
        model_slug = safe_name(model_name)
        dataset_slug = safe_name(dataset_name)

        run_dir = output_root / model_slug / dataset_slug
        run_dir.mkdir(parents=True, exist_ok=True)
        prompts_root = run_dir / "prompts"
        prompts_root.mkdir(parents=True, exist_ok=True)

        resolved = build_resolved_run_config(
            base_cfg,
            model_cfg,
            dataset_cfg,
            attrs,
            dimreds,
            chosen_device,
        )
        with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved, f, sort_keys=False, allow_unicode=True)
        save_json_v2(run_dir / "attr_index.json", attr_index)
        save_json_v2(run_dir / "dimred_index.json", dimred_index)

        prompts = load_prompts(dataset_cfg, args.max_prompts)
        print(f"[{run_i}/{len(combos)}] model={model_name} dataset={dataset_name} prompts={len(prompts)}")

        # BAUTEIL A: Generation-Komponenten einmal pro Kombination laden.
        # Erwarteter Output:
        # - hf_model / tokenizer, die in der Prompt-Schleife wiederverwendet werden.
        hf_model, hf_tokenizer = load_generation_components(
            model_name=model_name,
            device=chosen_device,
        )
        first_attr_name = attrs[0]["name"]
        inseq_model = inseq.load_model(
            model_name,
            attribution_method=first_attr_name,
            device=chosen_device,
        )
        if chosen_device == "mps":
            inseq_model.model = stabilize_model_for_metrics(inseq_model.model)
        can_switch_runtime = hasattr(inseq_model, "load_attribution_method")

        # LOOP 2: Prompt-Ebene.
        # Erwarteter Output:
        # - Pro Prompt ein eigener Ordner mit debug/result Dateien.
        for prompt_idx, prompt in enumerate(prompts):
            prompt_dir = prompts_root / f"prompt_{prompt_idx:03d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)

            debug_payload: Dict[str, Any] = {
                "prompt_idx": prompt_idx,
                "prompt": prompt,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "device": chosen_device,
                "status": "scaffold_only",
                "notes": [],
            }
            generation_cfg = dict(resolved.get("generation", {}) or {})
            seed_gen = _generation_seed(base_seed, prompt_idx)

            source_ids, generated_ids, full_text = generate_output_text(
                model=hf_model,
                tokenizer=hf_tokenizer,
                prompt=prompt,
                generation_cfg=generation_cfg,
                seed=seed_gen,
            )
            source_len = int(source_ids.shape[0])
            full_len = int(generated_ids.shape[0])
            t_gen = full_len - source_len

            # Shape sanity check nach Generation.
            _assert_1d_ids("source_ids", source_ids)
            _assert_1d_ids("generated_ids", generated_ids)
            if full_len <= source_len:
                raise ValueError("generated_ids must be longer than source_ids (need T_gen > 0)")

            debug_payload["generation"] = {
                "seed_gen": seed_gen,
                "generation_cfg": generation_cfg,
                "source_len": source_len,
                "full_len": full_len,
                "t_gen": t_gen,
                "source_text": hf_tokenizer.decode(generated_ids[:source_len], skip_special_tokens=False),
                "continuation_text": hf_tokenizer.decode(generated_ids[source_len:], skip_special_tokens=False),
                "full_text_preview": full_text[:120],
            }
            # Ultra-low-RAM: nach BAUTEIL A nur generated_ids + source_len behalten.
            gc.collect()
            clear_device_cache(chosen_device)

            # LOOP 3: Attribution-Methoden.
            # Erwarteter Output:
            # - Pro attribution method ein raw_target Tensor.
            for a_tag, a_cfg in attr_index.items():
                attr_name = a_cfg["name"]
                attr_idx = int(a_cfg["index"])
                attr_params = _prepare_attr_params_for_device(attr_name, a_cfg.get("params", {}), chosen_device)
                seed_attr = _attribution_seed(base_seed, prompt_idx, attr_idx)
                if can_switch_runtime or attr_name == first_attr_name:
                    model_for_attr = inseq_model
                    temp_model_loaded = False
                else:
                    model_for_attr = inseq.load_model(
                        model_name,
                        attribution_method=attr_name,
                        device=chosen_device,
                    )
                    if chosen_device == "mps":
                        model_for_attr.model = stabilize_model_for_metrics(model_for_attr.model)
                    temp_model_loaded = True

                if chosen_device == "mps" and _needs_mps_fp32_for_attr(attr_name):
                    model_for_attr.model = stabilize_model_for_metrics(model_for_attr.model)

                try:
                    set_global_seed(seed_attr)
                    attr_out = get_raw_targets_v2(
                        inseq_model=model_for_attr,
                        prompt=prompt,
                        generated_ids=generated_ids,
                        source_len=source_len,
                        attr_name=attr_name,
                        attr_params=attr_params,
                    )
                except Exception as exc:
                    debug_payload.setdefault("attr_debug", {})[a_tag] = {
                        "attr_name": attr_name,
                        "seed_attr": seed_attr,
                        "status": "failed",
                        "error": str(exc),
                    }
                    if _is_known_value_zeroing_incompatibility(attr_name, exc):
                        note = (
                            "Skipping value_zeroing: incompatible with current "
                            "inseq/transformers stack for this model output format."
                        )
                        debug_payload["attr_debug"][a_tag]["status"] = "skipped_incompatible"
                        debug_payload["attr_debug"][a_tag]["note"] = note
                        print(f"[warn] prompt_{prompt_idx:03d} attr={a_tag}: {note}", flush=True)
                    else:
                        print(
                            f"[warn] prompt_{prompt_idx:03d} attr={a_tag} failed: {exc}",
                            flush=True,
                        )
                    if temp_model_loaded:
                        del model_for_attr
                    gc.collect()
                    clear_device_cache(chosen_device)
                    continue

                raw_target = attr_out.raw_target
                # Shape sanity check direkt nach Attribution.
                _assert_raw_target_shape(
                     raw_target=raw_target,
                     source_len=source_len,
                     generated_ids=generated_ids,
                )
                debug_payload.setdefault("attr_debug", {})[a_tag] = {
                    "attr_name": attr_name,
                    "seed_attr": seed_attr,
                    "raw_target_shape": list(raw_target.shape),
                    "source_ids_debug_len": int(attr_out.source_ids_debug.shape[0]),
                    "target_ids_debug_len": int(attr_out.target_ids_debug.shape[0]),
                }
                nan_stats = raw_target_nan_stats(raw_target=raw_target, source_len=source_len)
                debug_payload["attr_debug"][a_tag]["nan_stats"] = nan_stats
                print(
                    f"[prompt_{prompt_idx:03d}] attr={a_tag} raw_target_shape={tuple(raw_target.shape)} "
                    f"global_nan_ratio={nan_stats['global_nan_ratio']:.6f} "
                    f"active_nan_ratio={nan_stats['active_nan_ratio']:.6f}",
                    flush=True,
                )
                baseline_l2 = l2_norm_over_dim(raw_target)
                _assert_importance_shape(
                    importance_map=baseline_l2,
                    source_len=source_len,
                    generated_ids=generated_ids,
                )
                metrics_model = stabilize_model_for_metrics(hf_model)
                seed_metrics_baseline = _baseline_metrics_seed(base_seed, prompt_idx, attr_idx)
                baseline_results = compute_soft_norm_metrics_v3(
                    metrics_model,
                    source_ids,
                    generated_ids[source_len:],
                    baseline_l2,
                    seed=seed_metrics_baseline,
                )
                save_baseline_result_v2(prompt_dir, a_tag, baseline_results)
                debug_payload["attr_debug"][a_tag]["baseline"] = {
                    "seed_attr": seed_attr,
                    "seed_metrics_baseline": seed_metrics_baseline,
                    "path_json": f"{a_tag}_baseline.json",
                    "path_steps_csv": f"{a_tag}_baseline_steps.csv",
                    "soft_ns_mean": float(baseline_results.soft_ns_mean),
                    "soft_nc_mean": float(baseline_results.soft_nc_mean),
                    "mean_kept_tokens_R": float(baseline_results.mean_kept_tokens_R),
                    "mean_kept_tokens_notR": float(baseline_results.mean_kept_tokens_notR),
                    "rationale_size_mode": baseline_results.rationale_size_mode,
                    "warnings": list(baseline_results.warnings),
                }

                # LOOP 4: DimRed-Methoden.
                # Erwarteter Output:
                # - Pro DimRed eine importance_map (L_total, T_gen)
                for d_tag, d_cfg in dimred_index.items():
                    dimred_name = d_cfg["name"]
                    dimred_idx = int(d_cfg["index"])
                    dimred_params = dict(d_cfg.get("params", {}) or {})
                    seed_metrics_dimred = _dimred_metrics_seed(base_seed, prompt_idx, attr_idx, dimred_idx)

                    dimred_map = reduce_raw_target(
                        raw_target,
                        dimred_name,
                        dimred_params,
                        seed=seed_metrics_dimred,
                    )
                    _assert_importance_shape(
                        importance_map=dimred_map,
                        source_len=source_len,
                        generated_ids=generated_ids,
                    )
                    results = compute_soft_norm_metrics_v3(
                        metrics_model,
                        source_ids,
                        generated_ids[source_len:],
                        dimred_map,
                        seed=seed_metrics_dimred,
                    )
                    save_dimred_result_v2(prompt_dir, a_tag, d_tag, results)

                    debug_payload["attr_debug"][a_tag].setdefault("dimred", {})[d_tag] = {
                        "dimred_name": dimred_name,
                        "seed_metrics_dimred": seed_metrics_dimred,
                        "dimred_map_shape": list(dimred_map.shape),
                        "path_json": f"{a_tag}_dimred_{d_tag}.json",
                        "path_steps_csv": f"{a_tag}_dimred_{d_tag}_steps.csv",
                        "soft_ns_mean": float(results.soft_ns_mean),
                        "soft_nc_mean": float(results.soft_nc_mean),
                        "mean_kept_tokens_R": float(results.mean_kept_tokens_R),
                        "mean_kept_tokens_notR": float(results.mean_kept_tokens_notR),
                        "rationale_size_mode": results.rationale_size_mode,
                        "warnings": list(results.warnings),
                        "status": "saved",
                    }

                # RAM-Hygiene pro Attribution-Methode:
                # Hier gezielt aufraeumen, damit keine grossen Tensoren im Speicher verbleiben.
                del attr_out
                del raw_target
                del baseline_l2
                del metrics_model
                if temp_model_loaded:
                    del model_for_attr
                gc.collect()
                clear_device_cache(chosen_device)

            save_json_v2(prompt_dir / "debug.json", debug_payload)

        del inseq_model
        del hf_model
        del hf_tokenizer
        gc.collect()
        clear_device_cache(chosen_device)


if __name__ == "__main__":
    main()
