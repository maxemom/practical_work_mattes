from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pretty_name(name: str) -> str:
    text = (name or "").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def _sem(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if clean.shape[0] <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(int(clean.shape[0])))


def _collect_combo_files(output_dir: Path) -> List[Path]:
    combo_files: List[Path] = []
    for path in output_dir.rglob("*.json"):
        if path.name in {"attr_index.json", "dimred_index.json", "run_meta.json"}:
            continue
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if "combo_key" in payload and "prompts" in payload:
            combo_files.append(path)
    return sorted(combo_files)


def _collect_prompt_dirs(output_dir: Path) -> List[Path]:
    prompt_dirs: List[Path] = []
    for path in output_dir.rglob("prompt.json"):
        if path.parent.name.startswith("prompt_") and path.parent.parent.name == "prompts":
            prompt_dirs.append(path.parent)
    return sorted(prompt_dirs)


def _build_long_form_from_prompt_dirs(output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for prompt_dir in _collect_prompt_dirs(output_dir):
        prompt_payload = _load_json(prompt_dir / "prompt.json")
        for path in sorted(prompt_dir.glob("*.json")):
            if path.name in {"prompt.json", "error.json", "debug.json"}:
                continue
            method = _load_json(path)
            if "combo_key" not in method:
                continue
            rows.append(
                {
                    "model_name": prompt_payload.get("model_name"),
                    "dataset_name": prompt_payload.get("dataset_name"),
                    "attribution": method.get("attribution_name"),
                    "dimred": method.get("dimred_name"),
                    "combo_key": method.get("combo_key"),
                    "prompt_idx": prompt_payload.get("prompt_idx"),
                    "skipped": bool(method.get("skipped", False)),
                    "skip_reason": method.get("skip_reason"),
                    "soft_ns_mean": method.get("soft_ns_mean"),
                    "soft_nc_mean": method.get("soft_nc_mean"),
                    "final_sufficiency_mean": method.get("final_sufficiency_mean"),
                    "final_comprehensiveness_mean": method.get("final_comprehensiveness_mean"),
                }
            )
    return pd.DataFrame(rows)


def build_long_form(output_dir: Path) -> pd.DataFrame:
    prompt_dirs = _collect_prompt_dirs(output_dir)
    if prompt_dirs:
        return _build_long_form_from_prompt_dirs(output_dir)

    rows: List[Dict[str, Any]] = []
    for path in _collect_combo_files(output_dir):
        payload = _load_json(path)
        for prompt_entry in payload.get("prompts", []):
            method = prompt_entry.get("method_result", {})
            rows.append(
                {
                    "model_name": payload.get("model_name"),
                    "dataset_name": payload.get("dataset_name"),
                    "attribution": payload.get("attribution_name"),
                    "dimred": payload.get("dimred_name"),
                    "combo_key": payload.get("combo_key"),
                    "prompt_idx": prompt_entry.get("prompt_idx"),
                    "skipped": bool(method.get("skipped", False)),
                    "skip_reason": method.get("skip_reason"),
                    "soft_ns_mean": method.get("soft_ns_mean"),
                    "soft_nc_mean": method.get("soft_nc_mean"),
                    "final_sufficiency_mean": method.get("final_sufficiency_mean"),
                    "final_comprehensiveness_mean": method.get("final_comprehensiveness_mean"),
                }
            )
    return pd.DataFrame(rows)


def _plot_metric_bars(run_df: pd.DataFrame, analysis_dir: Path, metric: str) -> None:
    grouped = (
        run_df[~run_df["skipped"]]
        .groupby(["model_name", "dataset_name", "attribution", "dimred"], as_index=False)
        .agg(
            mean_value=(metric, "mean"),
            sem_value=(metric, _sem),
        )
    )

    for (model_name, dataset_name, attribution), group in grouped.groupby(["model_name", "dataset_name", "attribution"]):
        labels = group["dimred"].tolist()
        values = group["mean_value"].astype(float).to_numpy()
        errors = group["sem_value"].astype(float).to_numpy()

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.25), 5), constrained_layout=True)
        x = np.arange(len(labels))
        ax.bar(x, values, yerr=errors, capsize=4, color="#0B6E4F", edgecolor="#1F2933")
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_name(label) for label in labels], rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} | {_pretty_name(attribution)} | {model_name} | {dataset_name}")
        ax.grid(axis="y", alpha=0.25)
        file_name = f"{model_name}__{dataset_name}__{attribution}__{metric}_bar.png"
        fig.savefig(analysis_dir / file_name, dpi=220)
        plt.close(fig)


def _plot_synergy_heatmap(run_df: pd.DataFrame, analysis_dir: Path) -> None:
    clean = run_df[~run_df["skipped"]].copy()
    if clean.empty:
        return

    for (model_name, dataset_name), dataset_group in clean.groupby(["model_name", "dataset_name"]):
        rows: List[Dict[str, Any]] = []
        for attribution, attr_group in dataset_group.groupby("attribution"):
            baseline_rows = attr_group[attr_group["dimred"] == "baseline"]
            if baseline_rows.empty:
                continue
            baseline_map = baseline_rows.set_index("prompt_idx")[
                ["final_sufficiency_mean", "final_comprehensiveness_mean"]
            ]
            for dimred, dim_group in attr_group.groupby("dimred"):
                if dimred == "baseline":
                    continue
                merged = dim_group.set_index("prompt_idx")[
                    ["final_sufficiency_mean", "final_comprehensiveness_mean"]
                ].join(
                    baseline_map,
                    how="inner",
                    rsuffix="_baseline",
                )
                if merged.empty:
                    continue
                synergy = 0.5 * (
                    (merged["final_sufficiency_mean"] - merged["final_sufficiency_mean_baseline"]).mean()
                    + (merged["final_comprehensiveness_mean"] - merged["final_comprehensiveness_mean_baseline"]).mean()
                )
                rows.append({"attribution": attribution, "dimred": dimred, "synergy": float(synergy)})

        if not rows:
            continue

        synergy_df = pd.DataFrame(rows)
        pivot = synergy_df.pivot(index="attribution", columns="dimred", values="synergy")

        fig, ax = plt.subplots(
            figsize=(max(7, 1.2 * len(pivot.columns)), max(4, 0.8 * len(pivot.index))),
            constrained_layout=True,
        )
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlGn")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([_pretty_name(name) for name in pivot.columns], rotation=30, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([_pretty_name(name) for name in pivot.index])
        ax.set_title(f"DimRed synergy vs baseline | {model_name} | {dataset_name}")

        for y, attr_name in enumerate(pivot.index):
            for x, dim_name in enumerate(pivot.columns):
                value = pivot.loc[attr_name, dim_name]
                label = "NA" if pd.isna(value) else f"{float(value):.3f}"
                ax.text(x, y, label, ha="center", va="center", color="#102A43")

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Synergy score")
        fig.savefig(analysis_dir / f"{model_name}__{dataset_name}__synergy_heatmap.png", dpi=220)
        plt.close(fig)


def _holm_correct(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values("p_value", ascending=True).reset_index(drop=True)
    m = len(ordered)
    adjusted: List[float] = []
    running_max = 0.0
    for idx, row in ordered.iterrows():
        raw = float(row["p_value"]) * float(m - idx)
        running_max = max(running_max, raw)
        adjusted.append(min(1.0, running_max))
    ordered["p_value_holm"] = adjusted
    ordered["significant"] = ordered["p_value_holm"] < 0.05
    return ordered


def _safe_wilcoxon(x: pd.Series, y: pd.Series) -> float:
    if len(x) == 0:
        return 1.0
    try:
        return float(wilcoxon(x, y, alternative="greater", zero_method="wilcox").pvalue)
    except Exception:
        return 1.0


def run_statistics(run_df: pd.DataFrame, analysis_dir: Path) -> pd.DataFrame:
    clean = run_df[~run_df["skipped"]].copy()
    rows: List[Dict[str, Any]] = []
    metrics = ["final_sufficiency_mean", "final_comprehensiveness_mean"]

    for (model_name, dataset_name, attribution), group in clean.groupby(["model_name", "dataset_name", "attribution"]):
        baseline = group[group["dimred"] == "baseline"][["prompt_idx", "final_sufficiency_mean", "final_comprehensiveness_mean"]]
        if baseline.empty:
            continue
        baseline = baseline.rename(
            columns={
                "final_sufficiency_mean": "baseline_final_sufficiency_mean",
                "final_comprehensiveness_mean": "baseline_final_comprehensiveness_mean",
            }
        )

        for metric_name in metrics:
            base_col = f"baseline_{metric_name}"
            for dimred, dim_group in group.groupby("dimred"):
                if dimred == "baseline":
                    continue
                merged = dim_group[["prompt_idx", metric_name]].merge(
                    baseline[["prompt_idx", base_col]],
                    on="prompt_idx",
                    how="inner",
                )
                if merged.empty:
                    continue
                effect = float((merged[metric_name] - merged[base_col]).mean())
                p_value = _safe_wilcoxon(merged[metric_name], merged[base_col])
                rows.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "attribution": attribution,
                        "dimred": dimred,
                        "metric": metric_name,
                        "paired_prompts": int(merged.shape[0]),
                        "effect_direction": effect,
                        "p_value": p_value,
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model_name",
                "dataset_name",
                "attribution",
                "dimred",
                "metric",
                "paired_prompts",
                "effect_direction",
                "p_value",
                "p_value_holm",
                "significant",
            ]
        )

    stats_df = pd.DataFrame(rows)
    corrected = []
    for _, group in stats_df.groupby(["model_name", "dataset_name", "attribution", "metric"]):
        corrected.append(_holm_correct(group))
    final_df = pd.concat(corrected, ignore_index=True)
    final_df.to_csv(analysis_dir / "statistical_tests.csv", index=False)
    return final_df


def interpret_outputs(output_dir: str = "outputs") -> Dict[str, Any]:
    output_path = ROOT / output_dir
    analysis_dir = output_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    long_form = build_long_form(output_path)
    long_form.to_csv(analysis_dir / "long_form.csv", index=False)
    if long_form.empty:
        return {"rows": 0, "analysis_dir": str(analysis_dir)}

    summary = (
        long_form[~long_form["skipped"]]
        .groupby(["model_name", "dataset_name", "attribution", "dimred"], as_index=False)
        .agg(
            soft_ns_mean=("soft_ns_mean", "mean"),
            soft_nc_mean=("soft_nc_mean", "mean"),
            final_sufficiency_mean=("final_sufficiency_mean", "mean"),
            final_comprehensiveness_mean=("final_comprehensiveness_mean", "mean"),
            prompt_count=("prompt_idx", "nunique"),
        )
    )
    summary.to_csv(analysis_dir / "summary.csv", index=False)

    for metric in ["final_sufficiency_mean", "final_comprehensiveness_mean"]:
        _plot_metric_bars(long_form, analysis_dir, metric)
    _plot_synergy_heatmap(long_form, analysis_dir)
    stats_df = run_statistics(long_form, analysis_dir)

    return {
        "rows": int(long_form.shape[0]),
        "summary_rows": int(summary.shape[0]),
        "stats_rows": int(stats_df.shape[0]),
        "analysis_dir": str(analysis_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpret combo-level experiment outputs and generate plots.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()
    report = interpret_outputs(args.output_dir)
    print(report)


if __name__ == "__main__":
    main()
