from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pwm.utils_base import load_yaml
from pwm.utils_pipeline import safe_name


@dataclass
class RunRecords:
    run_dir: Path
    model_name: str
    dataset_name: str
    attr_index: Dict[str, Dict[str, Any]]
    dimred_index: Dict[str, Dict[str, Any]]
    records: pd.DataFrame

    @property
    def run_label(self) -> str:
        return f"{safe_name(self.model_name)}__{safe_name(self.dataset_name)}"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pretty_name(name: str) -> str:
    text = (name or "").replace("_", " ").strip()
    text = " ".join(part for part in text.split() if part)
    mapping = {
        "input x gradient": "Input X Gradient",
        "gradient shap": "Gradient SHAP",
        "kernel pca": "Kernel PCA",
        "pca": "PCA",
        "ica": "ICA",
        "nmf": "NMF",
        "deeplift": "DeepLift",
        "lxt": "LXT",
        "value zeroing": "Value Zeroing",
        "baseline": "Baseline",
    }
    return mapping.get(text.lower(), text.title())


def set_paper_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.grid": False,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _std(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean.shape[0] <= 1:
        return 0.0
    return float(clean.std(ddof=1))


def _collect_run_dirs(output_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for run_meta in output_root.rglob("run_meta.json"):
        run_dirs.append(run_meta.parent)
    return sorted(set(run_dirs))


def _metadata_name(index_payload: Dict[str, Dict[str, Any]], tag: str, fallback: str) -> str:
    if tag in index_payload:
        return str(index_payload[tag].get("name", fallback))
    return fallback


def _metadata_params(index_payload: Dict[str, Dict[str, Any]], tag: str, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if tag in index_payload:
        return dict(index_payload[tag].get("params", {}) or {})
    return dict(fallback or {})


def _load_promptwise_records(run_dir: Path, attr_index: Dict[str, Dict[str, Any]], dimred_index: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    prompt_root = run_dir / "prompts"
    if not prompt_root.exists():
        return pd.DataFrame()

    for prompt_dir in sorted(prompt_root.glob("prompt_*")):
        prompt_json = prompt_dir / "prompt.json"
        if not prompt_json.exists():
            continue
        prompt_payload = _load_json(prompt_json)
        for path in sorted(prompt_dir.glob("*.json")):
            if path.name in {"prompt.json", "error.json", "debug.json"}:
                continue
            payload = _load_json(path)
            if "combo_key" not in payload:
                continue
            attr_tag = str(payload.get("attribution_tag") or safe_name(payload.get("attribution_name", "attr")))
            dimred_tag = str(payload.get("dimred_tag") or safe_name(payload.get("dimred_name", "dimred")))
            rows.append(
                {
                    "prompt_idx": int(prompt_payload.get("prompt_idx", 0)),
                    "prompt": prompt_payload.get("prompt"),
                    "generated_text": prompt_payload.get("generated_text"),
                    "source_ids": list(prompt_payload.get("source_ids", [])),
                    "total_ids": list(prompt_payload.get("total_ids", [])),
                    "generated_token_ids": list(prompt_payload.get("generated_token_ids", [])),
                    "generated_tokens": list(prompt_payload.get("generated_tokens", [])),
                    "source_len": int(prompt_payload.get("source_len", 0)),
                    "total_len": int(prompt_payload.get("total_len", 0)),
                    "model_name": prompt_payload.get("model_name"),
                    "dataset_name": prompt_payload.get("dataset_name"),
                    "combo_key": payload.get("combo_key"),
                    "attribution_tag": attr_tag,
                    "attribution_name": payload.get("attribution_name")
                    or _metadata_name(attr_index, attr_tag, attr_tag),
                    "attribution_params": dict(payload.get("attribution_params", {}) or _metadata_params(attr_index, attr_tag)),
                    "dimred_tag": dimred_tag,
                    "dimred_name": payload.get("dimred_name") or _metadata_name(dimred_index, dimred_tag, dimred_tag),
                    "dimred_params": dict(payload.get("dimred_params", {}) or _metadata_params(dimred_index, dimred_tag)),
                    "importance_scores": payload.get("importance_scores", []),
                    "soft_ns_per_token": list(payload.get("soft_ns_per_token", [])),
                    "soft_nc_per_token": list(payload.get("soft_nc_per_token", [])),
                    "final_sufficiency_per_token": list(payload.get("final_sufficiency_per_token", [])),
                    "final_comprehensiveness_per_token": list(payload.get("final_comprehensiveness_per_token", [])),
                    "soft_ns_mean": _safe_float(payload.get("soft_ns_mean")),
                    "soft_nc_mean": _safe_float(payload.get("soft_nc_mean")),
                    "final_sufficiency_mean": _safe_float(payload.get("final_sufficiency_mean")),
                    "final_comprehensiveness_mean": _safe_float(payload.get("final_comprehensiveness_mean")),
                    "target_pos": list(payload.get("target_pos", [])),
                    "target_token_ids": list(payload.get("target_token_ids", [])),
                    "target_token_texts": list(payload.get("target_token_texts", [])),
                    "warnings": list(payload.get("warnings", [])),
                    "skipped": bool(payload.get("skipped", False)),
                    "skip_reason": payload.get("skip_reason"),
                }
            )

    return pd.DataFrame(rows)


def _load_aggregate_records(run_dir: Path, attr_index: Dict[str, Dict[str, Any]], dimred_index: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name in {"run_meta.json", "attr_index.json", "dimred_index.json"}:
            continue
        payload = _load_json(path)
        if "combo_key" not in payload or "prompts" not in payload:
            continue
        for prompt_entry in payload.get("prompts", []):
            method = dict(prompt_entry.get("method_result", {}) or {})
            attr_tag = str(payload.get("attribution_tag") or method.get("attribution_tag") or safe_name(payload.get("attribution_name", "attr")))
            dimred_tag = str(payload.get("dimred_tag") or method.get("dimred_tag") or safe_name(payload.get("dimred_name", "dimred")))
            rows.append(
                {
                    "prompt_idx": int(prompt_entry.get("prompt_idx", 0)),
                    "prompt": prompt_entry.get("prompt"),
                    "generated_text": prompt_entry.get("generated_text"),
                    "source_ids": list(prompt_entry.get("source_ids", [])),
                    "total_ids": list(prompt_entry.get("total_ids", [])),
                    "generated_token_ids": list(prompt_entry.get("generated_token_ids", [])),
                    "generated_tokens": list(prompt_entry.get("generated_tokens", [])),
                    "source_len": int(prompt_entry.get("source_len", 0)),
                    "total_len": int(prompt_entry.get("total_len", 0)),
                    "model_name": payload.get("model_name"),
                    "dataset_name": payload.get("dataset_name"),
                    "combo_key": payload.get("combo_key"),
                    "attribution_tag": attr_tag,
                    "attribution_name": payload.get("attribution_name")
                    or method.get("attribution_name")
                    or _metadata_name(attr_index, attr_tag, attr_tag),
                    "attribution_params": dict(payload.get("attribution_params", {}) or method.get("attribution_params", {}) or _metadata_params(attr_index, attr_tag)),
                    "dimred_tag": dimred_tag,
                    "dimred_name": payload.get("dimred_name")
                    or method.get("dimred_name")
                    or _metadata_name(dimred_index, dimred_tag, dimred_tag),
                    "dimred_params": dict(payload.get("dimred_params", {}) or method.get("dimred_params", {}) or _metadata_params(dimred_index, dimred_tag)),
                    "importance_scores": method.get("importance_scores", []),
                    "soft_ns_per_token": list(method.get("soft_ns_per_token", [])),
                    "soft_nc_per_token": list(method.get("soft_nc_per_token", [])),
                    "final_sufficiency_per_token": list(method.get("final_sufficiency_per_token", [])),
                    "final_comprehensiveness_per_token": list(method.get("final_comprehensiveness_per_token", [])),
                    "soft_ns_mean": _safe_float(method.get("soft_ns_mean")),
                    "soft_nc_mean": _safe_float(method.get("soft_nc_mean")),
                    "final_sufficiency_mean": _safe_float(method.get("final_sufficiency_mean")),
                    "final_comprehensiveness_mean": _safe_float(method.get("final_comprehensiveness_mean")),
                    "target_pos": list(method.get("target_pos", [])),
                    "target_token_ids": list(method.get("target_token_ids", [])),
                    "target_token_texts": list(method.get("target_token_texts", [])),
                    "warnings": list(method.get("warnings", [])),
                    "skipped": bool(method.get("skipped", False)),
                    "skip_reason": method.get("skip_reason"),
                }
            )

    return pd.DataFrame(rows)


def load_run_records(run_dir: Path) -> RunRecords:
    run_meta_path = run_dir / "run_meta.json"
    run_meta = _load_json(run_meta_path) if run_meta_path.exists() else {}
    attr_index = _load_json(run_dir / "attr_index.json") if (run_dir / "attr_index.json").exists() else {}
    dimred_index = _load_json(run_dir / "dimred_index.json") if (run_dir / "dimred_index.json").exists() else {}

    promptwise = _load_promptwise_records(run_dir, attr_index, dimred_index)
    records = promptwise if not promptwise.empty else _load_aggregate_records(run_dir, attr_index, dimred_index)

    model_name = str(run_meta.get("model_name") or run_dir.parent.name)
    dataset_name = str(run_meta.get("dataset_name") or run_dir.name)
    if not records.empty:
        model_name = str(records["model_name"].dropna().iloc[0])
        dataset_name = str(records["dataset_name"].dropna().iloc[0])

    return RunRecords(
        run_dir=run_dir,
        model_name=model_name,
        dataset_name=dataset_name,
        attr_index=attr_index,
        dimred_index=dimred_index,
        records=records,
    )


def _selected_prompt_indices(
    available_prompt_indices: Sequence[int],
    prompt_idx: int | None = None,
    prompt_range: tuple[int, int] | None = None,
) -> list[int]:
    available = sorted({int(idx) for idx in available_prompt_indices})
    if not available:
        return []
    if prompt_idx is not None and prompt_range is not None:
        raise ValueError("Use either --prompt-idx or --prompt-range, not both.")
    if prompt_idx is not None:
        if int(prompt_idx) not in available:
            raise ValueError(f"Prompt {prompt_idx} not available. Available: {available}")
        return [int(prompt_idx)]
    if prompt_range is not None:
        start, end = int(prompt_range[0]), int(prompt_range[1])
        if end < start:
            raise ValueError("Prompt range must satisfy START <= END.")
        selected = [idx for idx in available if start <= idx <= end]
        if not selected:
            raise ValueError(f"No prompts found in range [{start}, {end}]. Available: {available}")
        return selected
    return available


def _prompt_selection_label(selected_prompt_indices: Sequence[int], total_prompt_count: int) -> str:
    selected = sorted(int(idx) for idx in selected_prompt_indices)
    if not selected:
        return "no_prompts"
    if len(selected) == total_prompt_count:
        return f"all_prompts_n{len(selected)}"
    if len(selected) == 1:
        return f"prompt_{selected[0]:03d}"
    return f"prompts_{selected[0]:03d}_{selected[-1]:03d}_n{len(selected)}"


def _prompt_selection_title(selected_prompt_indices: Sequence[int], total_prompt_count: int) -> str:
    selected = sorted(int(idx) for idx in selected_prompt_indices)
    if len(selected) == total_prompt_count:
        return f"all prompts (n={len(selected)})"
    if len(selected) == 1:
        return f"prompt {selected[0]:03d}"
    return f"prompts {selected[0]:03d}-{selected[-1]:03d} (n={len(selected)})"


def _query_matches(query: str | None, *, tag: str, name: str) -> bool:
    if query is None:
        return False
    norm_query = safe_name(query)
    return norm_query in {safe_name(tag), safe_name(name)}


def _resolve_attr_tag(run: RunRecords, query: str | None) -> str:
    available = [(str(tag), str(meta.get("name", tag))) for tag, meta in run.attr_index.items()]
    if not available and not run.records.empty:
        available = [
            (str(row["attribution_tag"]), str(row["attribution_name"]))
            for _, row in run.records[["attribution_tag", "attribution_name"]].drop_duplicates().iterrows()
        ]
    if not available:
        raise ValueError("No attribution methods available in run data.")

    if query is None:
        for tag, name in available:
            if safe_name(name) == "saliency":
                return tag
        return available[0][0]

    matches = [tag for tag, name in available if _query_matches(query, tag=tag, name=name)]
    if not matches:
        raise ValueError(f"Unknown attribution selection '{query}'.")
    if len(matches) > 1:
        raise ValueError(f"Attribution selection '{query}' is ambiguous: {matches}")
    return matches[0]


def _available_attr_tags(run: RunRecords, records: pd.DataFrame) -> list[str]:
    available = _ordered_attr_tags(run, records)
    return [tag for tag in available if tag in set(records["attribution_tag"].astype(str).tolist())]


def _resolve_dimred_tag(run: RunRecords, query: str) -> str:
    available = [("baseline", "baseline")]
    available.extend((str(tag), str(meta.get("name", tag))) for tag, meta in run.dimred_index.items())
    deduped: list[tuple[str, str]] = []
    seen_tags: set[str] = set()
    for tag, name in available:
        if tag in seen_tags:
            continue
        deduped.append((tag, name))
        seen_tags.add(tag)

    matches = [tag for tag, name in deduped if _query_matches(query, tag=tag, name=name)]
    if not matches:
        raise ValueError(f"Unknown dimred selection '{query}'.")
    if len(matches) > 1:
        raise ValueError(f"Dimred selection '{query}' is ambiguous: {matches}")
    return matches[0]


def _available_dimred_tags(run: RunRecords, records: pd.DataFrame) -> list[str]:
    available = _ordered_dimred_tags(run, records)
    return [tag for tag in available if tag in set(records["dimred_tag"].astype(str).tolist())]


def _ordered_attr_tags(run: RunRecords, records: pd.DataFrame) -> list[str]:
    ordered = list(run.attr_index.keys())
    present = records["attribution_tag"].dropna().astype(str).drop_duplicates().tolist()
    for tag in present:
        if tag not in ordered:
            ordered.append(tag)
    return ordered


def _ordered_dimred_tags(run: RunRecords, records: pd.DataFrame) -> list[str]:
    ordered = []
    if not records.empty and "baseline" in records["dimred_tag"].astype(str).tolist():
        ordered.append("baseline")
    for tag in run.dimred_index.keys():
        if tag not in ordered:
            ordered.append(tag)
    present = records["dimred_tag"].dropna().astype(str).drop_duplicates().tolist()
    for tag in present:
        if tag not in ordered:
            ordered.append(tag)
    return ordered


def _dimred_display_label(run: RunRecords, dimred_tag: str, dimred_name: str) -> str:
    if dimred_tag == "baseline" or safe_name(dimred_name) == "baseline":
        return "Baseline"
    params = dict(run.dimred_index.get(dimred_tag, {}).get("params", {}) or {})
    if "n_components" in params:
        return f"{_pretty_name(dimred_name)}\n(k={params['n_components']})"
    return _pretty_name(dimred_name)


def _extract_n_components(value: Any) -> int | None:
    try:
        if value is None:
            return None
        component_count = int(value)
    except Exception:
        return None
    if component_count <= 0:
        return None
    return component_count


def _dimred_n_components(run: RunRecords, dimred_tag: str, dimred_params: Dict[str, Any] | None = None) -> int | None:
    params = dict(dimred_params or {})
    component_count = _extract_n_components(params.get("n_components"))
    if component_count is not None:
        return component_count

    index_params = dict(run.dimred_index.get(dimred_tag, {}).get("params", {}) or {})
    component_count = _extract_n_components(index_params.get("n_components"))
    if component_count is not None:
        return component_count

    match = re.search(r"(?:^|_)n_components_(\d+)(?:_|$)", safe_name(dimred_tag))
    if match:
        return _extract_n_components(match.group(1))
    return None


def summarize_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()

    usable = records[~records["skipped"]].copy()
    if usable.empty:
        return pd.DataFrame()

    return (
        usable.groupby(
            ["attribution_tag", "attribution_name", "dimred_tag", "dimred_name"],
            as_index=False,
        )
        .agg(
            prompt_count=("prompt_idx", "nunique"),
            soft_suff_mean=("soft_ns_mean", "mean"),
            soft_suff_std=("soft_ns_mean", _std),
            soft_comp_mean=("soft_nc_mean", "mean"),
            soft_comp_std=("soft_nc_mean", _std),
            final_suff_mean=("final_sufficiency_mean", "mean"),
            final_suff_std=("final_sufficiency_mean", _std),
            final_comp_mean=("final_comprehensiveness_mean", "mean"),
            final_comp_std=("final_comprehensiveness_mean", _std),
        )
    )


def _sign_test_counts(values: Sequence[float], zero_tol: float = 1e-12) -> tuple[int, int, int]:
    clean = [float(value) for value in values if np.isfinite(value)]
    positives = sum(1 for value in clean if value > zero_tol)
    negatives = sum(1 for value in clean if value < -zero_tol)
    return positives, negatives, positives + negatives


def _two_sided_sign_test_pvalue(values: Sequence[float], zero_tol: float = 1e-12) -> float:
    positives, negatives, trials = _sign_test_counts(values, zero_tol=zero_tol)
    if trials <= 0:
        return 1.0
    tail = min(positives, negatives)
    cumulative = 0.0
    denom = 2**trials
    for successes in range(tail + 1):
        cumulative += math.comb(trials, successes) / denom
    return float(min(1.0, 2.0 * cumulative))


def _baseline_direction(delta_mean: float, p_value: float, alpha: float) -> str:
    if not np.isfinite(delta_mean):
        return "na"
    if not np.isfinite(p_value) or p_value >= alpha:
        return "no_clear_difference"
    if delta_mean > 0.0:
        return "better"
    if delta_mean < 0.0:
        return "worse"
    return "no_clear_difference"


def compute_baseline_comparison_stats(records: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()

    usable = records[~records["skipped"]].copy()
    if usable.empty:
        return pd.DataFrame()

    metric_specs = [
        ("soft_ns_mean", "soft_suff"),
        ("soft_nc_mean", "soft_comp"),
        ("final_sufficiency_mean", "final_suff"),
        ("final_comprehensiveness_mean", "final_comp"),
    ]

    rows: list[dict[str, Any]] = []
    for _, attr_subset in usable.groupby("attribution_tag", sort=False):
        baseline = attr_subset[attr_subset["dimred_tag"].astype(str) == "baseline"].copy()
        if baseline.empty:
            continue

        baseline = (
            baseline.sort_values("prompt_idx")
            .drop_duplicates(subset=["prompt_idx"], keep="first")
            .rename(columns={metric_name: f"{metric_name}_baseline" for metric_name, _ in metric_specs})
        )

        for _, method_subset in attr_subset.groupby("dimred_tag", sort=False):
            method_subset = method_subset.sort_values("prompt_idx").drop_duplicates(subset=["prompt_idx"], keep="first")
            paired = method_subset.merge(
                baseline[
                    [
                        "prompt_idx",
                        *[f"{metric_name}_baseline" for metric_name, _ in metric_specs],
                    ]
                ],
                on="prompt_idx",
                how="inner",
            )
            if paired.empty:
                continue

            sample_row = method_subset.iloc[0]
            row: dict[str, Any] = {
                "attribution_tag": str(sample_row["attribution_tag"]),
                "attribution_name": str(sample_row["attribution_name"]),
                "dimred_tag": str(sample_row["dimred_tag"]),
                "dimred_name": str(sample_row["dimred_name"]),
                "paired_prompt_count": int(paired["prompt_idx"].nunique()),
            }

            for metric_name, prefix in metric_specs:
                method_values = pd.to_numeric(paired[metric_name], errors="coerce").astype(float)
                baseline_values = pd.to_numeric(paired[f"{metric_name}_baseline"], errors="coerce").astype(float)
                deltas = (method_values - baseline_values).to_numpy(dtype=float)
                positives, negatives, nonzero_count = _sign_test_counts(deltas)
                p_value = _two_sided_sign_test_pvalue(deltas)
                delta_mean = float(np.nanmean(deltas)) if deltas.size else float("nan")
                delta_median = float(np.nanmedian(deltas)) if deltas.size else float("nan")
                row[f"{prefix}_method_mean"] = float(np.nanmean(method_values.to_numpy(dtype=float)))
                row[f"{prefix}_baseline_mean"] = float(np.nanmean(baseline_values.to_numpy(dtype=float)))
                row[f"{prefix}_delta_mean"] = delta_mean
                row[f"{prefix}_delta_median"] = delta_median
                row[f"{prefix}_p_value"] = p_value
                row[f"{prefix}_positive_pairs"] = int(positives)
                row[f"{prefix}_negative_pairs"] = int(negatives)
                row[f"{prefix}_nonzero_pairs"] = int(nonzero_count)
                row[f"{prefix}_direction"] = _baseline_direction(delta_mean, p_value, alpha)
                row[f"{prefix}_test"] = "paired_sign_test"

            rows.append(row)

    return pd.DataFrame(rows)


def _dynamic_limits(values: Iterable[float], include_zero: bool = False) -> tuple[float, float]:
    arr = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    if arr.size == 0:
        return (-0.1, 0.1) if include_zero else (0.0, 1.0)
    lower = float(arr.min())
    upper = float(arr.max())
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    if math.isclose(lower, upper):
        pad = 0.1 if math.isclose(lower, 0.0) else abs(lower) * 0.15
        return lower - pad, upper + pad
    pad = max((upper - lower) * 0.12, 0.02)
    return lower - pad, upper + pad


def plot_baseline_bars(
    run: RunRecords,
    summary_df: pd.DataFrame,
    selected_prompt_indices: Sequence[int],
    out_dir: Path,
    dimred_query: str = "baseline",
) -> Path | None:
    if summary_df.empty:
        return None

    dimred_tag = _resolve_dimred_tag(run, dimred_query)
    subset = summary_df[summary_df["dimred_tag"] == dimred_tag].copy()
    if subset.empty:
        return None

    ordered_tags = _ordered_attr_tags(run, subset)
    subset["attr_order"] = subset["attribution_tag"].map({tag: idx for idx, tag in enumerate(ordered_tags)})
    subset = subset.sort_values("attr_order")

    methods = [_pretty_name(name) for name in subset["attribution_name"].tolist()]
    suff_values = subset["soft_suff_mean"].to_numpy(dtype=float)
    comp_values = subset["soft_comp_mean"].to_numpy(dtype=float)
    suff_errors = subset["soft_suff_std"].to_numpy(dtype=float)
    comp_errors = subset["soft_comp_std"].to_numpy(dtype=float)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(methods)))
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(max(8.5, len(methods) * 1.15), 4.1))
    for ax, values, errors, ylabel, show_zero in (
        (axes[0], suff_values, suff_errors, "Soft Suff", True),
        (axes[1], comp_values, comp_errors, "Soft Comp", False),
    ):
        ax.bar(x, values, yerr=errors, capsize=4, color=colors, edgecolor="#1F2933", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_xlabel("Methods")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.18)
        if show_zero:
            ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(*_dynamic_limits(values, include_zero=show_zero))

    selection_title = _prompt_selection_title(selected_prompt_indices, run.records["prompt_idx"].nunique())
    dimred_name = subset["dimred_name"].iloc[0]
    fig.suptitle(
        f"{_pretty_name(dimred_name)} comparison | {run.model_name} | {run.dataset_name} | {selection_title}",
        y=0.98,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)

    path = out_dir / f"{run.run_label}__{_prompt_selection_label(selected_prompt_indices, run.records['prompt_idx'].nunique())}__{dimred_tag}__bars.png"
    save_figure(fig, path)
    return path


def _significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value <= 0.001:
        return "***"
    if p_value <= 0.01:
        return "**"
    if p_value <= 0.05:
        return "*"
    return ""


def _format_p_value(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def _resolve_diverging_limits(matrix: np.ndarray) -> tuple[float, float]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return (-1.0, 1.0)
    max_abs = float(np.max(np.abs(finite)))
    if math.isclose(max_abs, 0.0):
        max_abs = 1e-6
    return (-max_abs, max_abs)


def plot_baseline_stat_heatmaps(
    run: RunRecords,
    stats_df: pd.DataFrame,
    selected_prompt_indices: Sequence[int],
    out_dir: Path,
    *,
    alpha: float = 0.05,
) -> Path | None:
    if stats_df.empty:
        return None

    attr_tags = _ordered_attr_tags(run, stats_df)
    dimred_tags = _ordered_dimred_tags(run, stats_df)
    attr_labels = {}
    for tag in attr_tags:
        matches = stats_df[stats_df["attribution_tag"] == tag]
        label = matches["attribution_name"].iloc[0] if not matches.empty else run.attr_index.get(tag, {}).get("name", tag)
        attr_labels[tag] = _pretty_name(str(label))

    dimred_labels = {}
    for tag in dimred_tags:
        matches = stats_df[stats_df["dimred_tag"] == tag]
        dimred_name = matches["dimred_name"].iloc[0] if not matches.empty else run.dimred_index.get(tag, {}).get("name", tag)
        dimred_labels[tag] = _dimred_display_label(run, tag, str(dimred_name))

    attr_to_y = {tag: idx for idx, tag in enumerate(attr_tags)}
    dimred_to_x = {tag: idx for idx, tag in enumerate(dimred_tags)}
    metric_configs = [
        ("soft_suff", "Soft Suff Delta vs Baseline"),
        ("soft_comp", "Soft Comp Delta vs Baseline"),
    ]

    delta_matrices: dict[str, np.ndarray] = {}
    pvalue_matrices: dict[str, np.ndarray] = {}
    count_matrices: dict[str, np.ndarray] = {}
    for prefix, _ in metric_configs:
        delta_matrices[prefix] = np.full((len(attr_tags), len(dimred_tags)), np.nan, dtype=float)
        pvalue_matrices[prefix] = np.full((len(attr_tags), len(dimred_tags)), np.nan, dtype=float)
        count_matrices[prefix] = np.full((len(attr_tags), len(dimred_tags)), np.nan, dtype=float)

    for _, row in stats_df.iterrows():
        y = attr_to_y[str(row["attribution_tag"])]
        x = dimred_to_x[str(row["dimred_tag"])]
        for prefix, _ in metric_configs:
            delta_matrices[prefix][y, x] = float(row.get(f"{prefix}_delta_mean", float("nan")))
            pvalue_matrices[prefix][y, x] = float(row.get(f"{prefix}_p_value", float("nan")))
            count_matrices[prefix][y, x] = float(row.get(f"{prefix}_nonzero_pairs", float("nan")))

    fig, axes = plt.subplots(1, 2, figsize=(max(9.8, len(dimred_tags) * 1.25), max(4.8, len(attr_tags) * 0.68)))
    selection_title = _prompt_selection_title(selected_prompt_indices, run.records["prompt_idx"].nunique())

    for ax, (prefix, title) in zip(axes, metric_configs):
        mean_matrix = delta_matrices[prefix]
        p_matrix = pvalue_matrices[prefix]
        n_matrix = count_matrices[prefix]
        lower, upper = _resolve_diverging_limits(mean_matrix)
        norm = mcolors.TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
        im = ax.imshow(mean_matrix, aspect="auto", cmap="RdBu_r", norm=norm)
        ax.set_xticks(np.arange(len(dimred_tags)))
        ax.set_xticklabels([dimred_labels[tag] for tag in dimred_tags], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(attr_tags)))
        ax.set_yticklabels([attr_labels[tag] for tag in attr_tags])
        ax.set_title(title)

        ax.set_xticks(np.arange(-0.5, len(dimred_tags), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(attr_tags), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        for y in range(mean_matrix.shape[0]):
            for x in range(mean_matrix.shape[1]):
                delta_value = mean_matrix[y, x]
                p_value = p_matrix[y, x]
                nonzero_pairs = n_matrix[y, x]
                dimred_tag = dimred_tags[x]

                if dimred_tag == "baseline":
                    text = "ref\np=1.000"
                elif not np.isfinite(delta_value):
                    text = "NA"
                else:
                    stars = _significance_stars(p_value)
                    p_text = _format_p_value(p_value)
                    n_text = "0" if not np.isfinite(nonzero_pairs) else str(int(nonzero_pairs))
                    text = f"{delta_value:+.2f}{stars}\np={p_text}\nn={n_text}"

                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=7.8,
                    color=_heatmap_text_color(delta_value, norm),
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Delta vs baseline")

    fig.suptitle(
        f"Paired sign test vs L2 baseline | {run.model_name} | {run.dataset_name} | {selection_title} | alpha={alpha:.2f}",
        y=0.98,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    path = out_dir / (
        f"{run.run_label}__{_prompt_selection_label(selected_prompt_indices, run.records['prompt_idx'].nunique())}"
        "__baseline_stats_heatmaps.png"
    )
    save_figure(fig, path)
    return path


def _resolve_color_limits(matrix: np.ndarray, lower: float | None, upper: float | None) -> tuple[float, float]:
    if lower is not None and upper is not None:
        return float(lower), float(upper)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return (0.0, 1.0)
    if lower is None:
        lower = float(finite.min())
    if upper is None:
        upper = float(finite.max())
    if math.isclose(float(lower), float(upper)):
        upper = float(upper) + 1e-6
    return float(lower), float(upper)


def _heatmap_text_color(value: float, norm: mcolors.Normalize) -> str:
    if not np.isfinite(value):
        return "#111827"
    return "white" if norm(value) >= 0.6 else "#111827"


def plot_metric_heatmaps(
    run: RunRecords,
    summary_df: pd.DataFrame,
    selected_prompt_indices: Sequence[int],
    out_dir: Path,
    *,
    suff_vmin: float | None = None,
    suff_vmax: float | None = None,
    comp_vmin: float | None = None,
    comp_vmax: float | None = None,
) -> Path | None:
    if summary_df.empty:
        return None

    attr_tags = _ordered_attr_tags(run, summary_df)
    dimred_tags = _ordered_dimred_tags(run, summary_df)
    attr_labels = {}
    for tag in attr_tags:
        matches = summary_df[summary_df["attribution_tag"] == tag]
        label = matches["attribution_name"].iloc[0] if not matches.empty else run.attr_index.get(tag, {}).get("name", tag)
        attr_labels[tag] = _pretty_name(str(label))

    dimred_labels = {}
    for tag in dimred_tags:
        matches = summary_df[summary_df["dimred_tag"] == tag]
        dimred_name = matches["dimred_name"].iloc[0] if not matches.empty else run.dimred_index.get(tag, {}).get("name", tag)
        dimred_labels[tag] = _dimred_display_label(run, tag, str(dimred_name))

    attr_to_y = {tag: idx for idx, tag in enumerate(attr_tags)}
    dimred_to_x = {tag: idx for idx, tag in enumerate(dimred_tags)}
    suff_mean = np.full((len(attr_tags), len(dimred_tags)), np.nan, dtype=float)
    suff_std = np.full_like(suff_mean, np.nan)
    comp_mean = np.full_like(suff_mean, np.nan)
    comp_std = np.full_like(suff_mean, np.nan)

    for _, row in summary_df.iterrows():
        y = attr_to_y[str(row["attribution_tag"])]
        x = dimred_to_x[str(row["dimred_tag"])]
        suff_mean[y, x] = float(row["soft_suff_mean"])
        suff_std[y, x] = float(row["soft_suff_std"])
        comp_mean[y, x] = float(row["soft_comp_mean"])
        comp_std[y, x] = float(row["soft_comp_std"])

    fig, axes = plt.subplots(1, 2, figsize=(max(9.8, len(dimred_tags) * 1.2), max(4.6, len(attr_tags) * 0.64)))
    selection_title = _prompt_selection_title(selected_prompt_indices, run.records["prompt_idx"].nunique())

    configs = [
        (axes[0], suff_mean, suff_std, "Soft Suff", "Greens", suff_vmin, suff_vmax),
        (axes[1], comp_mean, comp_std, "Soft Comp", "Reds", comp_vmin, comp_vmax),
    ]

    for ax, mean_matrix, std_matrix, title, cmap_name, lower, upper in configs:
        vmin, vmax = _resolve_color_limits(mean_matrix, lower, upper)
        im = ax.imshow(mean_matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(dimred_tags)))
        ax.set_xticklabels([dimred_labels[tag] for tag in dimred_tags], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(attr_tags)))
        ax.set_yticklabels([attr_labels[tag] for tag in attr_tags])
        ax.set_title(title)

        ax.set_xticks(np.arange(-0.5, len(dimred_tags), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(attr_tags), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for y in range(mean_matrix.shape[0]):
            for x in range(mean_matrix.shape[1]):
                value = mean_matrix[y, x]
                deviation = std_matrix[y, x]
                text = "NA" if not np.isfinite(value) else f"{value:.2f}\n+/-{deviation:.2f}"
                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=_heatmap_text_color(value, norm),
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title)

    fig.suptitle(f"{run.model_name} | {run.dataset_name} | {selection_title}", y=0.98)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    path = out_dir / f"{run.run_label}__{_prompt_selection_label(selected_prompt_indices, run.records['prompt_idx'].nunique())}__metric_heatmaps.png"
    save_figure(fig, path)
    return path


def _ordered_component_dimred_names(run: RunRecords, records: pd.DataFrame) -> list[str]:
    ordered: list[str] = []
    for _, meta in run.dimred_index.items():
        name = str(meta.get("name", ""))
        if not name or safe_name(name) == "baseline":
            continue
        params = dict(meta.get("params", {}) or {})
        if _extract_n_components(params.get("n_components")) is None:
            continue
        if name not in ordered:
            ordered.append(name)

    if not records.empty:
        for name in records["dimred_name"].dropna().astype(str).drop_duplicates().tolist():
            if safe_name(name) == "baseline":
                continue
            if name not in ordered:
                ordered.append(name)
    return ordered


def _best_n_component_dimred_tags(run: RunRecords, records: pd.DataFrame) -> list[str]:
    if records.empty:
        return []

    ordered_tags = _ordered_dimred_tags(run, records)
    present_tags = set(records["dimred_tag"].dropna().astype(str).tolist())
    selected_tags: set[str] = set()

    usable = records[~records["skipped"]].copy()
    if not usable.empty:
        usable["dimred_n_components"] = usable.apply(
            lambda row: _dimred_n_components(
                run,
                str(row["dimred_tag"]),
                dict(row.get("dimred_params", {}) or {}),
            ),
            axis=1,
        )
        component_records = usable[
            usable["dimred_n_components"].notna()
            & (usable["dimred_name"].astype(str).map(safe_name) != "baseline")
        ].copy()

        if not component_records.empty:
            component_records["dimred_n_components"] = component_records["dimred_n_components"].astype(int)
            component_summary = (
                component_records.groupby(["dimred_name", "dimred_tag", "dimred_n_components"], as_index=False)
                .agg(
                    soft_suff_mean=("soft_ns_mean", "mean"),
                    soft_comp_mean=("soft_nc_mean", "mean"),
                    sample_count=("prompt_idx", "count"),
                )
            )
            component_summary["soft_suff_rank"] = component_summary.groupby("dimred_name")["soft_suff_mean"].rank(
                method="min",
                ascending=False,
                na_option="bottom",
            )
            component_summary["soft_comp_rank"] = component_summary.groupby("dimred_name")["soft_comp_mean"].rank(
                method="min",
                ascending=False,
                na_option="bottom",
            )
            component_summary["best_rank"] = (
                component_summary["soft_suff_rank"] + component_summary["soft_comp_rank"]
            ) / 2.0
            component_summary = component_summary.sort_values(
                ["dimred_name", "best_rank", "dimred_n_components", "sample_count", "dimred_tag"],
                ascending=[True, True, True, False, True],
            )
            best_rows = component_summary.drop_duplicates(subset=["dimred_name"], keep="first")
            selected_tags.update(best_rows["dimred_tag"].astype(str).tolist())

    for tag in ordered_tags:
        matches = records[records["dimred_tag"].astype(str) == tag]
        if matches.empty:
            continue
        sample = matches.iloc[0]
        dimred_name = str(sample["dimred_name"])
        n_components = _dimred_n_components(run, str(tag), dict(sample.get("dimred_params", {}) or {}))
        if safe_name(dimred_name) == "baseline" or n_components is None:
            selected_tags.add(str(tag))

    return [tag for tag in ordered_tags if tag in present_tags and tag in selected_tags]


def filter_to_best_n_components_per_dimred(run: RunRecords, records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    selected_tags = _best_n_component_dimred_tags(run, records)
    if not selected_tags:
        return records.iloc[0:0].copy()
    return records[records["dimred_tag"].astype(str).isin(selected_tags)].copy()


def plot_n_components_comparison(
    run: RunRecords,
    records: pd.DataFrame,
    selected_prompt_indices: Sequence[int],
    out_dir: Path,
) -> Path | None:
    if records.empty:
        return None

    usable = records[~records["skipped"]].copy()
    if usable.empty:
        return None

    usable["dimred_n_components"] = usable.apply(
        lambda row: _dimred_n_components(
            run,
            str(row["dimred_tag"]),
            dict(row.get("dimred_params", {}) or {}),
        ),
        axis=1,
    )
    usable = usable[usable["dimred_n_components"].notna()].copy()
    usable = usable[usable["dimred_name"].astype(str).map(safe_name) != "baseline"].copy()
    if usable.empty:
        return None

    usable["dimred_n_components"] = usable["dimred_n_components"].astype(int)
    component_summary = (
        usable.groupby(["dimred_name", "dimred_n_components"], as_index=False)
        .agg(
            sample_count=("prompt_idx", "count"),
            prompt_count=("prompt_idx", "nunique"),
            attribution_count=("attribution_tag", "nunique"),
            soft_suff_mean=("soft_ns_mean", "mean"),
            soft_suff_std=("soft_ns_mean", _std),
            soft_comp_mean=("soft_nc_mean", "mean"),
            soft_comp_std=("soft_nc_mean", _std),
        )
    )
    if component_summary.empty:
        return None

    dimred_names = [
        name
        for name in _ordered_component_dimred_names(run, usable)
        if name in set(component_summary["dimred_name"].astype(str).tolist())
    ]
    if not dimred_names:
        dimred_names = component_summary["dimred_name"].astype(str).drop_duplicates().tolist()

    component_values = sorted(component_summary["dimred_n_components"].astype(int).drop_duplicates().tolist())
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(1, len(dimred_names))))
    selection_title = _prompt_selection_title(selected_prompt_indices, run.records["prompt_idx"].nunique())

    fig, axes = plt.subplots(1, 2, figsize=(max(10.5, len(component_values) * 1.2 + len(dimred_names) * 0.7), 4.7))
    metric_configs = [
        (axes[0], "soft_suff_mean", "soft_suff_std", "Soft Suff", True),
        (axes[1], "soft_comp_mean", "soft_comp_std", "Soft Comp", False),
    ]

    for ax, mean_col, std_col, ylabel, show_zero in metric_configs:
        all_values: list[float] = []
        for color, dimred_name in zip(colors, dimred_names):
            subset = component_summary[component_summary["dimred_name"].astype(str) == dimred_name].sort_values("dimred_n_components")
            x = subset["dimred_n_components"].to_numpy(dtype=int)
            y = subset[mean_col].to_numpy(dtype=float)
            yerr = subset[std_col].to_numpy(dtype=float)
            all_values.extend(float(value) for value in y if np.isfinite(value))
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                markersize=5,
                capsize=3,
                linewidth=1.8,
                color=color,
                label=_pretty_name(str(dimred_name)),
            )

        ax.set_xlabel("n_components")
        ax.set_ylabel(ylabel)
        ax.set_xticks(component_values)
        ax.grid(axis="both", alpha=0.18)
        if show_zero:
            ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(*_dynamic_limits(all_values, include_zero=show_zero))
        ax.set_title(ylabel)

    axes[1].legend(title="DimRed", loc="best", frameon=True)
    fig.suptitle(
        f"n_components comparison | {run.model_name} | {run.dataset_name} | {selection_title} | averaged over attribution methods",
        y=0.98,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.84)

    path = out_dir / (
        f"{run.run_label}__{_prompt_selection_label(selected_prompt_indices, run.records['prompt_idx'].nunique())}"
        "__n_components_comparison.png"
    )
    save_figure(fig, path)
    return path


def _resolve_tokenizer(model_name: str, local_files_only: bool = False) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    except Exception:
        return None


def _decode_total_tokens(tokenizer: Any | None, total_ids: Sequence[int]) -> list[str]:
    if tokenizer is None:
        return [str(int(token_id)) for token_id in total_ids]
    decoded: list[str] = []
    for token_id in total_ids:
        try:
            decoded.append(str(tokenizer.decode([int(token_id)], skip_special_tokens=False)))
        except Exception:
            decoded.append(str(int(token_id)))
    return decoded


def _resolve_token_prompt_idx(selected_prompt_indices: Sequence[int], token_prompt_idx: int | None) -> int:
    selected = sorted(int(idx) for idx in selected_prompt_indices)
    if not selected:
        raise ValueError("No selected prompts available for token plot.")
    if token_prompt_idx is None:
        return selected[0]
    if int(token_prompt_idx) not in selected:
        raise ValueError(f"Token plot prompt {token_prompt_idx} is not in the selected prompt subset {selected}.")
    return int(token_prompt_idx)


def _resolve_token_target_pos(record: pd.Series, token_target_pos: int | None) -> tuple[int, int]:
    target_positions = [int(value) for value in (record.get("target_pos") or [])]
    if not target_positions:
        raise ValueError("Selected record does not contain target_pos values.")
    chosen_target_pos = target_positions[-1] if token_target_pos is None else int(token_target_pos)
    if chosen_target_pos not in target_positions:
        raise ValueError(f"Target position {chosen_target_pos} not available. Available: {target_positions}")
    return chosen_target_pos, target_positions.index(chosen_target_pos)


def _matrix_from_payload(payload: Any) -> np.ndarray:
    rows: list[list[float]] = []
    for row in list(payload or []):
        rows.append([float("nan") if value is None else float(value) for value in list(row)])
    return np.asarray(rows, dtype=float)


def _token_layout(tokens: Sequence[str]) -> tuple[list[tuple[float, float]], float]:
    layout: list[tuple[float, float]] = []
    cursor = 0.0
    for token in tokens:
        display_token = str(token) if str(token) else "<empty>"
        width = max(1.15, len(display_token) * 0.24 + 0.8)
        center = cursor + width / 2.0
        layout.append((center, width))
        cursor += width + 0.32
    return layout, cursor


def plot_token_attribution_rows(
    run: RunRecords,
    records: pd.DataFrame,
    selected_prompt_indices: Sequence[int],
    out_dir: Path,
    *,
    token_prompt_idx: int | None = None,
    token_attr_query: str | None = None,
    token_dimred_queries: Sequence[str] | None = None,
    token_target_pos: int | None = None,
    local_files_only: bool = False,
) -> Path | None:
    if records.empty:
        return None

    chosen_prompt_idx = _resolve_token_prompt_idx(selected_prompt_indices, token_prompt_idx)
    chosen_attr_tag = _resolve_attr_tag(run, token_attr_query)
    prompt_records = records[
        (records["prompt_idx"] == chosen_prompt_idx)
        & (records["attribution_tag"] == chosen_attr_tag)
        & (~records["skipped"])
    ].copy()
    if prompt_records.empty:
        return None

    if token_dimred_queries:
        chosen_dimred_tags = [_resolve_dimred_tag(run, query) for query in token_dimred_queries]
    else:
        chosen_dimred_tags = _ordered_dimred_tags(run, prompt_records)
    prompt_records = prompt_records[prompt_records["dimred_tag"].isin(chosen_dimred_tags)].copy()
    if prompt_records.empty:
        return None
    prompt_records["dimred_order"] = prompt_records["dimred_tag"].map({tag: idx for idx, tag in enumerate(chosen_dimred_tags)})
    prompt_records = prompt_records.sort_values("dimred_order")

    anchor_record = prompt_records.iloc[0]
    chosen_target_pos, column_index = _resolve_token_target_pos(anchor_record, token_target_pos)
    total_ids = list(anchor_record.get("total_ids", []))
    if not total_ids:
        return None

    tokenizer = _resolve_tokenizer(run.model_name, local_files_only=local_files_only)
    total_tokens = _decode_total_tokens(tokenizer, total_ids)
    visible_len = min(len(total_tokens), chosen_target_pos + 1)
    visible_tokens = total_tokens[:visible_len]
    layout, total_width = _token_layout(visible_tokens)

    score_arrays: list[np.ndarray] = []
    for _, row in prompt_records.iterrows():
        matrix = _matrix_from_payload(row.get("importance_scores"))
        if matrix.ndim != 2 or matrix.shape[1] <= column_index:
            scores = np.full((visible_len,), np.nan, dtype=float)
        else:
            scores = matrix[:visible_len, column_index]
        score_arrays.append(scores)

    non_target_scores = []
    for scores in score_arrays:
        for token_idx, score in enumerate(scores):
            if token_idx == chosen_target_pos:
                continue
            if np.isfinite(score):
                non_target_scores.append(abs(float(score)))
    vmax = max(non_target_scores) if non_target_scores else 1.0
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
    cmap = plt.cm.Purples

    row_gap = 2.25
    fig_height = max(3.2, len(prompt_records) * 1.65 + 0.9)
    fig_width = max(10.0, total_width * 0.31)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    max_y = row_gap * (len(prompt_records) - 1) + 1.65
    for row_idx, (_, row) in enumerate(prompt_records.iterrows()):
        base_y = max_y - row_idx * row_gap
        dimred_title = _dimred_display_label(run, str(row["dimred_tag"]), str(row["dimred_name"]))
        attr_title = _pretty_name(str(row["attribution_name"]))
        ax.text(0.0, base_y + 0.58, f"{dimred_title} ({attr_title})", ha="left", va="center", fontsize=11, fontweight="bold")

        scores = score_arrays[row_idx]
        for token_idx, ((center_x, width), token_text) in enumerate(zip(layout, visible_tokens)):
            score = scores[token_idx] if token_idx < scores.shape[0] else float("nan")
            if token_idx == chosen_target_pos:
                facecolor = "#A7E3A1"
                edgecolor = "#2E7D32"
            elif np.isfinite(score):
                facecolor = cmap(norm(abs(float(score))))
                edgecolor = "#C7CAD1"
            else:
                facecolor = "#F3F4F6"
                edgecolor = "#D1D5DB"

            label = "" if not np.isfinite(score) else f"{float(score):.2f}"
            ax.text(center_x, base_y + 0.16, label, ha="center", va="center", fontsize=8)
            ax.text(
                center_x,
                base_y - 0.22,
                str(token_text),
                ha="center",
                va="center",
                fontsize=10,
                bbox={
                    "boxstyle": "square,pad=0.25",
                    "facecolor": facecolor,
                    "edgecolor": edgecolor,
                    "linewidth": 0.8,
                },
            )

    ax.set_xlim(-0.35, total_width + 0.35)
    ax.set_ylim(-0.75, max_y + 1.1)
    selection_title = _prompt_selection_title(selected_prompt_indices, run.records["prompt_idx"].nunique())
    fig.suptitle(
        f"{run.model_name} | {run.dataset_name} | {selection_title} | prompt {chosen_prompt_idx:03d} | target_pos {chosen_target_pos}",
        y=0.98,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.87)

    path = out_dir / (
        f"{run.run_label}__prompt_{chosen_prompt_idx:03d}__{chosen_attr_tag}__target_{chosen_target_pos}__token_rows.png"
    )
    save_figure(fig, path)
    return path


def _filter_run_dirs(run_dirs: Sequence[Path], model_name: str | None, dataset_name: str | None) -> list[Path]:
    filtered: list[Path] = []
    for run_dir in run_dirs:
        model_slug = run_dir.parent.name
        dataset_slug = run_dir.name
        if model_name and safe_name(model_name) != model_slug and model_name != model_slug:
            continue
        if dataset_name and safe_name(dataset_name) != dataset_slug and dataset_name != dataset_slug:
            continue
        filtered.append(run_dir)
    return filtered


def _resolve_path_from_root(path_value: str | None, default_relative: str) -> Path:
    if path_value:
        path = Path(path_value)
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    return (ROOT / default_relative).resolve()


def _run_dirs_from_grid_config(output_root: Path, grid_path: Path) -> list[Path]:
    grid_cfg = load_yaml(grid_path)
    models = [str(item.get("name", "")) for item in list(grid_cfg.get("models", []))]
    datasets = [str(item.get("name", "")) for item in list(grid_cfg.get("datasets", []))]
    wanted = {
        (safe_name(model_name), safe_name(dataset_name))
        for model_name in models
        for dataset_name in datasets
        if model_name and dataset_name
    }
    if not wanted:
        return []

    run_dirs = _collect_run_dirs(output_root)
    return [
        run_dir
        for run_dir in run_dirs
        if (run_dir.parent.name, run_dir.name) in wanted
    ]


def create_all_plots(
    *,
    output_root: str = "outputs",
    plot_dir: str | None = None,
    grid_path: str | None = None,
    model_name: str | None = None,
    dataset_name: str | None = None,
    prompt_idx: int | None = None,
    prompt_range: tuple[int, int] | None = None,
    bar_dimred: str = "baseline",
    all_bar_dimreds: bool = True,
    skip_bar_plot: bool = False,
    skip_stat_plot: bool = False,
    skip_heatmap_plot: bool = False,
    skip_component_plot: bool = False,
    skip_token_plot: bool = False,
    token_attribution: str | None = None,
    token_dimreds: Sequence[str] | None = None,
    all_token_attributions: bool = True,
    token_prompt_idx: int | None = None,
    token_target_pos: int | None = None,
    local_files_only: bool = False,
    suff_vmin: float | None = None,
    suff_vmax: float | None = None,
    comp_vmin: float | None = None,
    comp_vmax: float | None = None,
    stat_alpha: float = 0.05,
) -> Dict[str, Any]:
    set_paper_plot_style()

    resolved_output_root = _resolve_path_from_root(output_root, "outputs")
    resolved_plot_root = (
        _resolve_path_from_root(plot_dir, "outputs/plots/create_plots")
        if plot_dir
        else resolved_output_root / "plots" / "create_plots"
    )
    resolved_plot_root.mkdir(parents=True, exist_ok=True)

    resolved_grid_path = None
    if grid_path:
        resolved_grid_path = _resolve_path_from_root(grid_path, grid_path)

    if resolved_grid_path is not None:
        run_dirs = _run_dirs_from_grid_config(resolved_output_root, resolved_grid_path)
        run_dirs = _filter_run_dirs(run_dirs, model_name, dataset_name)
    else:
        run_dirs = _filter_run_dirs(_collect_run_dirs(resolved_output_root), model_name, dataset_name)

    report: Dict[str, Any] = {
        "output_root": str(resolved_output_root),
        "plot_root": str(resolved_plot_root),
        "grid_path": str(resolved_grid_path) if resolved_grid_path is not None else None,
        "run_count": 0,
        "runs": [],
        "skipped": [],
    }

    if not run_dirs:
        print(f"[skip] no runs found below {resolved_output_root}")
        return report

    token_dimred_queries = [str(item).strip() for item in list(token_dimreds or []) if str(item).strip()] or None

    for run_dir in run_dirs:
        run = load_run_records(run_dir)
        if run.records.empty:
            print(f"[skip] no prompt-level records found in {run_dir}")
            report["skipped"].append({"run_dir": str(run_dir), "reason": "no_prompt_level_records"})
            continue

        try:
            selected_prompt_indices = _selected_prompt_indices(
                run.records["prompt_idx"].tolist(),
                prompt_idx=prompt_idx,
                prompt_range=prompt_range,
            )
        except ValueError as exc:
            print(f"[skip] {run.run_label}: {exc}")
            report["skipped"].append({"run_dir": str(run_dir), "reason": str(exc)})
            continue

        filtered_records = run.records[run.records["prompt_idx"].isin(selected_prompt_indices)].copy()
        summary_df = summarize_records(filtered_records)
        stats_df = compute_baseline_comparison_stats(filtered_records, alpha=stat_alpha)
        if summary_df.empty:
            print(f"[skip] {run.run_label}: no successful records after prompt filtering")
            report["skipped"].append({"run_dir": str(run_dir), "reason": "no_successful_records_after_prompt_filtering"})
            continue
        plot_records = filter_to_best_n_components_per_dimred(run, filtered_records)
        plot_summary_df = summarize_records(plot_records)
        plot_stats_df = compute_baseline_comparison_stats(plot_records, alpha=stat_alpha)

        run_plot_dir = resolved_plot_root / run.run_label
        run_plot_dir.mkdir(parents=True, exist_ok=True)
        selection_label = _prompt_selection_label(selected_prompt_indices, run.records["prompt_idx"].nunique())
        summary_df.to_csv(run_plot_dir / f"{run.run_label}__{selection_label}__summary.csv", index=False)
        if not stats_df.empty:
            stats_df.to_csv(run_plot_dir / f"{run.run_label}__{selection_label}__baseline_stats.csv", index=False)
        if not plot_summary_df.empty:
            plot_summary_df.to_csv(run_plot_dir / f"{run.run_label}__{selection_label}__best_n_components_summary.csv", index=False)
        if not plot_stats_df.empty:
            plot_stats_df.to_csv(run_plot_dir / f"{run.run_label}__{selection_label}__best_n_components_baseline_stats.csv", index=False)

        generated_paths: dict[str, list[str]] = {
            "bars": [],
            "stats": [],
            "heatmaps": [],
            "components": [],
            "tokens": [],
        }

        if not skip_bar_plot:
            try:
                bar_dimred_queries = [bar_dimred]
                if all_bar_dimreds:
                    bar_dimred_queries = _available_dimred_tags(run, plot_summary_df)
                for dimred_query in bar_dimred_queries:
                    path = plot_baseline_bars(
                        run=run,
                        summary_df=plot_summary_df,
                        selected_prompt_indices=selected_prompt_indices,
                        out_dir=run_plot_dir,
                        dimred_query=dimred_query,
                    )
                    if path is not None:
                        generated_paths["bars"].append(str(path))
            except Exception as exc:
                print(f"[warn] {run.run_label}: bar plot failed: {exc}")

        if not skip_stat_plot:
            try:
                path = plot_baseline_stat_heatmaps(
                    run=run,
                    stats_df=plot_stats_df,
                    selected_prompt_indices=selected_prompt_indices,
                    out_dir=run_plot_dir,
                    alpha=stat_alpha,
                )
                if path is not None:
                    generated_paths["stats"].append(str(path))
            except Exception as exc:
                print(f"[warn] {run.run_label}: baseline stats plot failed: {exc}")

        if not skip_heatmap_plot:
            try:
                path = plot_metric_heatmaps(
                    run=run,
                    summary_df=plot_summary_df,
                    selected_prompt_indices=selected_prompt_indices,
                    out_dir=run_plot_dir,
                    suff_vmin=suff_vmin,
                    suff_vmax=suff_vmax,
                    comp_vmin=comp_vmin,
                    comp_vmax=comp_vmax,
                )
                if path is not None:
                    generated_paths["heatmaps"].append(str(path))
            except Exception as exc:
                print(f"[warn] {run.run_label}: heatmap plot failed: {exc}")

        if not skip_component_plot:
            try:
                path = plot_n_components_comparison(
                    run=run,
                    records=filtered_records,
                    selected_prompt_indices=selected_prompt_indices,
                    out_dir=run_plot_dir,
                )
                if path is not None:
                    generated_paths["components"].append(str(path))
            except Exception as exc:
                print(f"[warn] {run.run_label}: n_components plot failed: {exc}")

        if not skip_token_plot:
            try:
                token_attr_queries = [token_attribution]
                if all_token_attributions:
                    token_attr_queries = _available_attr_tags(run, plot_records)
                for token_attr_query in token_attr_queries:
                    path = plot_token_attribution_rows(
                        run=run,
                        records=plot_records,
                        selected_prompt_indices=selected_prompt_indices,
                        out_dir=run_plot_dir,
                        token_prompt_idx=token_prompt_idx,
                        token_attr_query=token_attr_query,
                        token_dimred_queries=token_dimred_queries,
                        token_target_pos=token_target_pos,
                        local_files_only=local_files_only,
                    )
                    if path is not None:
                        generated_paths["tokens"].append(str(path))
            except Exception as exc:
                print(f"[warn] {run.run_label}: token plot failed: {exc}")

        print(f"[ok] wrote plots to {run_plot_dir}")
        report["runs"].append(
            {
                "run_dir": str(run_dir),
                "run_label": run.run_label,
                "plot_dir": str(run_plot_dir),
                "selected_prompt_indices": list(selected_prompt_indices),
                "generated_paths": generated_paths,
            }
        )

    report["run_count"] = len(report["runs"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Create comparison plots from normal outputs.")
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument("--grid", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--prompt-idx", type=int, default=None, help="Restrict plots to a single prompt.")
    parser.add_argument(
        "--prompt-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Restrict plots to an inclusive prompt range.",
    )
    parser.add_argument(
        "--bar-dimred",
        type=str,
        default="baseline",
        help="DimRed tag or method name used for the 1x2 bar comparison.",
    )
    parser.add_argument(
        "--all-bar-dimreds",
        action="store_true",
        help="Create one bar comparison for every available dimred variant.",
    )
    parser.add_argument("--skip-bar-plot", action="store_true")
    parser.add_argument("--skip-stat-plot", action="store_true")
    parser.add_argument("--skip-heatmap-plot", action="store_true")
    parser.add_argument("--skip-component-plot", action="store_true")
    parser.add_argument("--skip-token-plot", action="store_true")
    parser.add_argument("--stat-alpha", type=float, default=0.05)
    parser.add_argument(
        "--token-attribution",
        type=str,
        default=None,
        help="Attribution tag or name for the token visualization. Default: saliency if available.",
    )
    parser.add_argument(
        "--token-dimreds",
        type=str,
        default=None,
        help="Comma-separated dimred tags or names for token rows. Default: all available.",
    )
    parser.add_argument(
        "--all-token-attributions",
        action="store_true",
        help="Create token attribution plots for every available attribution method.",
    )
    parser.add_argument(
        "--token-prompt-idx",
        type=int,
        default=None,
        help="Prompt index used only for the token plot. Default: first selected prompt.",
    )
    parser.add_argument(
        "--token-target-pos",
        type=int,
        default=None,
        help="Target position to visualize in the token plot. Default: last available target position.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--suff-vmin", type=float, default=None)
    parser.add_argument("--suff-vmax", type=float, default=None)
    parser.add_argument("--comp-vmin", type=float, default=None)
    parser.add_argument("--comp-vmax", type=float, default=None)
    args = parser.parse_args()
    token_dimred_queries = None
    if args.token_dimreds:
        token_dimred_queries = [item.strip() for item in args.token_dimreds.split(",") if item.strip()]

    create_all_plots(
        output_root=args.output_root,
        plot_dir=args.plot_dir,
        grid_path=args.grid,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        prompt_idx=args.prompt_idx,
        prompt_range=tuple(args.prompt_range) if args.prompt_range else None,
        bar_dimred=args.bar_dimred,
        all_bar_dimreds=args.all_bar_dimreds,
        skip_bar_plot=args.skip_bar_plot,
        skip_stat_plot=args.skip_stat_plot,
        skip_heatmap_plot=args.skip_heatmap_plot,
        skip_component_plot=args.skip_component_plot,
        skip_token_plot=args.skip_token_plot,
        token_attribution=args.token_attribution,
        token_dimreds=token_dimred_queries,
        all_token_attributions=args.all_token_attributions,
        token_prompt_idx=args.token_prompt_idx,
        token_target_pos=args.token_target_pos,
        local_files_only=args.local_files_only,
        suff_vmin=args.suff_vmin,
        suff_vmax=args.suff_vmax,
        comp_vmin=args.comp_vmin,
        comp_vmax=args.comp_vmax,
        stat_alpha=args.stat_alpha,
    )


if __name__ == "__main__":
    main()
