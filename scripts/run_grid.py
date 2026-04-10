from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pwm.main_function import ExperimentRuntime, run_prompt_experiment
from pwm.typess import ComboAggregateResult, PromptCombinationRecord, PromptRunResult
from pwm.utils_base import load_yaml
from pwm.utils_pipeline import ALLOWED_ATTR_METHODS, ALLOWED_DIMRED_METHODS, build_attr_index, build_dimred_index, configure_runtime_warnings, disable_loading_verbosity, load_prompts, patch_lxt_transformers_compatibility, safe_name
from pwm.utils_results import save_json
from pwm.utils_runtime import build_resolved_run_config, resolve_device, resolve_model_dtype


def _version_or_na(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except Exception:
        return "n/a"


def _check_python_module(module_name: str) -> tuple[bool, str]:
    found = importlib.util.find_spec(module_name) is not None
    return found, "available" if found else "missing"


def check_environment(base_cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    modules = list(base_cfg.get("environment", {}).get("required_modules", []))
    reports: List[Dict[str, str]] = []
    if not modules:
        return reports

    print("=== Environment Check ===")
    for module_name in modules:
        ok, reason = _check_python_module(str(module_name))
        status = "OK" if ok else "MISSING"
        print(f"[env] {module_name}: {status} ({reason})")
        reports.append({"module": str(module_name), "status": status, "reason": reason})
    return reports


def _check_docker_files(base_cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    reports: List[Dict[str, str]] = []
    docker_cfg = base_cfg.get("docker", {})
    expected = list(docker_cfg.get("expected_files", []))
    if not expected:
        return reports

    print("=== Docker Check ===")
    for rel_path in expected:
        path = ROOT / rel_path
        ok = path.exists()
        status = "OK" if ok else "MISSING"
        print(f"[docker] {rel_path}: {status}")
        reports.append({"path": rel_path, "status": status})
    return reports


def _check_model_support(model_cfg: Dict[str, Any]) -> tuple[bool, str]:
    model_name = str(model_cfg.get("name", ""))
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(model_name)
        return True, "config_resolved"
    except Exception as exc:
        return False, str(exc)


def _check_dataset_support(dataset_cfg: Dict[str, Any], max_prompts: int | None = None) -> tuple[bool, str, List[str]]:
    try:
        prompts = load_prompts(dataset_cfg, max_prompts)
    except Exception as exc:
        return False, str(exc), []
    return True, f"{len(prompts)} prompts", prompts


def _check_attr_support(attr_cfg: Dict[str, Any]) -> tuple[bool, str]:
    name = str(attr_cfg.get("name", "")).lower()
    if name not in ALLOWED_ATTR_METHODS:
        return False, "not in allowed attribution set"
    if name == "lxt":
        ok, reason = _check_python_module("lxt")
        if not ok:
            return False, "missing python module 'lxt' (install LRP-eXplains-Transformers)"
        sub_ok, _ = _check_python_module("lxt.efficient")
        if not sub_ok:
            return False, "missing submodule 'lxt.efficient' (install LRP-eXplains-Transformers)"
        try:
            patch_lxt_transformers_compatibility()
            from lxt.efficient import monkey_patch

            del monkey_patch
        except Exception as exc:
            return False, f"lxt import failed after compatibility shims: {exc}"
    return True, "supported"


def _check_dimred_support(dimred_cfg: Dict[str, Any]) -> tuple[bool, str]:
    name = str(dimred_cfg.get("name", "")).lower()
    if name not in ALLOWED_DIMRED_METHODS:
        return False, "not in allowed dimred set"
    return True, "supported"


def _attach_tags(index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for tag, meta in index.items():
        tagged.append(
            {
                "tag": tag,
                "name": meta["name"],
                "params": dict(meta.get("params", {}) or {}),
                "index": int(meta.get("index", 0)),
            }
        )
    return tagged


def build_combo_aggregates(
    model_name: str,
    dataset_name: str,
    run_meta: Dict[str, Any],
    attrs: Iterable[Dict[str, Any]],
    dimreds: Iterable[Dict[str, Any]],
) -> Dict[str, ComboAggregateResult]:
    aggregates: Dict[str, ComboAggregateResult] = {}
    for attr_cfg in attrs:
        for dim_cfg in dimreds:
            combo_key = f"{attr_cfg['tag']}__{dim_cfg['tag']}"
            aggregates[combo_key] = ComboAggregateResult(
                combo_key=combo_key,
                model_name=model_name,
                dataset_name=dataset_name,
                attribution_tag=str(attr_cfg["tag"]),
                attribution_name=str(attr_cfg["name"]),
                attribution_params=dict(attr_cfg.get("params", {}) or {}),
                dimred_tag=str(dim_cfg["tag"]),
                dimred_name=str(dim_cfg["name"]),
                dimred_params=dict(dim_cfg.get("params", {}) or {}),
                run_meta=dict(run_meta),
            )
    return aggregates


def append_prompt_result(
    aggregates: Dict[str, ComboAggregateResult],
    prompt_result: PromptRunResult,
) -> None:
    for combo_key, method_result in prompt_result.combinations.items():
        if combo_key not in aggregates:
            continue
        aggregates[combo_key].prompts.append(
            PromptCombinationRecord(
                prompt_idx=prompt_result.prompt_idx,
                prompt=prompt_result.prompt,
                generated_text=prompt_result.generated_text,
                source_ids=list(prompt_result.source_ids),
                total_ids=list(prompt_result.total_ids),
                generated_token_ids=list(prompt_result.generated_token_ids),
                source_len=prompt_result.source_len,
                total_len=prompt_result.total_len,
                generated_tokens=list(prompt_result.generated_tokens),
                method_result=method_result,
            )
        )


def summarize_aggregate(aggregate: ComboAggregateResult) -> Dict[str, Any]:
    valid = [entry.method_result for entry in aggregate.prompts if not entry.method_result.skipped]
    skipped = [entry.method_result for entry in aggregate.prompts if entry.method_result.skipped]

    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    warnings = sorted(
        {
            warning
            for entry in aggregate.prompts
            for warning in entry.method_result.warnings
        }
    )

    summary = {
        "prompt_count": len(aggregate.prompts),
        "successful_prompt_count": len(valid),
        "skipped_prompt_count": len(skipped),
        "soft_ns_mean": _mean([item.soft_ns_mean for item in valid]),
        "soft_nc_mean": _mean([item.soft_nc_mean for item in valid]),
        "final_sufficiency_mean": _mean([item.final_sufficiency_mean for item in valid]),
        "final_comprehensiveness_mean": _mean([item.final_comprehensiveness_mean for item in valid]),
        "skip_reasons": sorted({item.skip_reason for item in skipped if item.skip_reason}),
        "warnings": warnings,
    }
    aggregate.summary = summary
    aggregate.warnings = warnings
    return summary


def _save_aggregate_file(run_dir: Path, aggregate: ComboAggregateResult) -> None:
    summarize_aggregate(aggregate)
    save_json(run_dir / f"{aggregate.combo_key}.json", aggregate)


def save_all_aggregates(run_dir: Path, aggregates: Dict[str, ComboAggregateResult]) -> None:
    for aggregate in aggregates.values():
        _save_aggregate_file(run_dir, aggregate)


def _prompt_dir(run_dir: Path, prompt_idx: int) -> Path:
    return run_dir / "prompts" / f"prompt_{prompt_idx:03d}"


def _build_prompt_payload(prompt_result: PromptRunResult) -> Dict[str, Any]:
    return {
        "prompt_idx": int(prompt_result.prompt_idx),
        "prompt": prompt_result.prompt,
        "model_name": prompt_result.model_name,
        "dataset_name": prompt_result.dataset_name,
        "generated_text": prompt_result.generated_text,
        "source_ids": list(prompt_result.source_ids),
        "total_ids": list(prompt_result.total_ids),
        "generated_token_ids": list(prompt_result.generated_token_ids),
        "generated_tokens": list(prompt_result.generated_tokens),
        "source_len": int(prompt_result.source_len),
        "total_len": int(prompt_result.total_len),
        "warnings": list(prompt_result.warnings),
    }


def _combo_result_file_name(method_result: Any) -> str:
    if str(method_result.dimred_tag) == "baseline":
        return f"{method_result.attribution_tag}_baseline.json"
    return f"{method_result.attribution_tag}_dimred_{method_result.dimred_tag}.json"


def save_prompt_outputs(run_dir: Path, prompt_result: PromptRunResult) -> None:
    prompt_dir = _prompt_dir(run_dir, prompt_result.prompt_idx)
    save_json(prompt_dir / "prompt.json", _build_prompt_payload(prompt_result))
    for method_result in prompt_result.combinations.values():
        save_json(prompt_dir / _combo_result_file_name(method_result), method_result)


def _print_support(kind: str, name: str, ok: bool, reason: str) -> None:
    status = "OK" if ok else "SKIP"
    print(f"[support] {kind}={name}: {status} ({reason})")


def run_grid(
    *,
    base_path: str = "configs/base.yaml",
    grid_path: str = "configs/grid.yaml",
    max_prompts: int | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    disable_loading_verbosity()
    configure_runtime_warnings()
    base_cfg = load_yaml(Path(base_path))
    grid_cfg = load_yaml(Path(grid_path))

    env_reports = check_environment(base_cfg)
    docker_reports = _check_docker_files(base_cfg)

    requested_device = str(base_cfg.get("runtime", {}).get("device", "auto"))
    device_report = resolve_device(requested_device)
    print(
        f"[runtime] requested_device={requested_device} chosen_device={device_report.chosen} "
        f"cuda={device_report.cuda_available} mps={device_report.mps_available}"
    )

    attr_index = build_attr_index(grid_cfg.get("attribution_functions", []))
    dimred_index = build_dimred_index(grid_cfg.get("dimensionality_reduction_methods", []))
    tagged_attrs = _attach_tags(attr_index)
    tagged_dimreds = _attach_tags(dimred_index)

    valid_attrs: List[Dict[str, Any]] = []
    valid_dimreds: List[Dict[str, Any]] = []

    for attr_cfg in tagged_attrs:
        ok, reason = _check_attr_support(attr_cfg)
        _print_support("attribution", str(attr_cfg["name"]), ok, reason)
        if ok:
            valid_attrs.append(attr_cfg)

    for dim_cfg in tagged_dimreds:
        ok, reason = _check_dimred_support(dim_cfg)
        _print_support("dimred", str(dim_cfg["name"]), ok, reason)
        if ok:
            valid_dimreds.append(dim_cfg)

    output_root = ROOT / str(base_cfg.get("paths", {}).get("output_dir", "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)

    run_report: Dict[str, Any] = {
        "environment": env_reports,
        "docker": docker_reports,
        "runs": [],
    }

    for model_cfg in grid_cfg.get("models", []):
        model_ok, model_reason = _check_model_support(model_cfg)
        _print_support("model", str(model_cfg.get("name", "")), model_ok, model_reason)
        if not model_ok:
            continue

        for dataset_cfg in grid_cfg.get("datasets", []):
            dataset_ok, dataset_reason, prompts = _check_dataset_support(dataset_cfg, max_prompts=max_prompts)
            _print_support("dataset", str(dataset_cfg.get("name", "")), dataset_ok, dataset_reason)
            if not dataset_ok or not valid_attrs or not valid_dimreds:
                continue

            model_name = str(model_cfg["name"])
            dataset_name = str(dataset_cfg["name"])
            resolved = build_resolved_run_config(
                base_cfg=base_cfg,
                model_cfg=model_cfg,
                dataset_cfg=dataset_cfg,
                attrs=valid_attrs,
                dimreds=valid_dimreds,
                chosen_device=device_report.chosen,
            )
            _, model_dtype_name = resolve_model_dtype(resolved)

            model_slug = safe_name(model_name)
            dataset_slug = safe_name(dataset_name)
            run_dir = output_root / model_slug / dataset_slug
            run_dir.mkdir(parents=True, exist_ok=True)

            with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(resolved, handle, sort_keys=False, allow_unicode=True)
            save_json(run_dir / "attr_index.json", attr_index)
            save_json(run_dir / "dimred_index.json", dimred_index)

            run_meta = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version.split()[0],
                "torch": _version_or_na("torch"),
                "transformers": _version_or_na("transformers"),
                "inseq": _version_or_na("inseq"),
                "scipy": _version_or_na("scipy"),
                "model_name": model_name,
                "dataset_name": dataset_name,
                "requested_device": requested_device,
                "device": device_report.chosen,
                "model_dtype": model_dtype_name,
            }
            save_json(run_dir / "run_meta.json", run_meta)

            print(f"[run] model={model_name} dataset={dataset_name} prompts={len(prompts)} dtype={model_dtype_name}")

            if dry_run:
                continue

            runtime = ExperimentRuntime.create(model_name, resolved, valid_attrs)
            try:
                for prompt_idx, prompt in enumerate(prompts):
                    print(f"[prompt] idx={prompt_idx} text_len={len(prompt)}")
                    prompt_result = run_prompt_experiment(
                        prompt=prompt,
                        model_name=model_name,
                        resolved_config=resolved,
                        attribution_methods=valid_attrs,
                        dimred_methods=valid_dimreds,
                        prompt_idx=prompt_idx,
                        dataset_name=dataset_name,
                        runtime=runtime,
                    )
                    save_prompt_outputs(run_dir, prompt_result)
                    print(
                        f"[save] prompt={prompt_idx} combo_jsons_saved={len(prompt_result.combinations)} dir={_prompt_dir(run_dir, prompt_idx)}",
                        flush=True,
                    )
            finally:
                runtime.close()

            run_report["runs"].append(
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "output_dir": str(run_dir),
                    "combination_count": len(valid_attrs) * len(valid_dimreds),
                    "prompt_count": len(prompts),
                }
            )

    return run_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the prompt -> attribution -> dimred -> metrics experiment grid.")
    parser.add_argument("--base", type=str, default="configs/base.yaml")
    parser.add_argument("--grid", type=str, default="configs/grid.yaml")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_grid(
        base_path=args.base,
        grid_path=args.grid,
        max_prompts=args.max_prompts,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
