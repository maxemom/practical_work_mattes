from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pretty_name(name: str) -> str:
    text = (name or "").strip().replace("_", " ")
    text = " ".join(part for part in text.split() if part)
    mapping = {
        "input x gradient": "Input x Gradient",
        "gradient shap": "Gradient SHAP",
        "kernel pca": "Kernel PCA",
        "pca": "PCA",
        "ica": "ICA",
        "nmf": "NMF",
        "deeplift": "DeepLift",
        "baseline": "Baseline",
    }
    return mapping.get(text.lower(), text.title())


def _sem(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    n = int(clean.shape[0])
    if n <= 1:
        return 0.0
    return float(clean.std(ddof=1) / math.sqrt(n))


def _run_label(run_dir: Path) -> str:
    dataset = run_dir.name
    model = run_dir.parent.name
    return f"{model}__{dataset}"


def _collect_records(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    attr_index = _load_json(run_dir / "attr_index.json")
    dimred_index = _load_json(run_dir / "dimred_index.json")
    prompt_dirs = sorted((run_dir / "prompts").glob("prompt_*"))

    rows: list[dict[str, Any]] = []
    for prompt_dir in prompt_dirs:
        prompt_idx = int(prompt_dir.name.split("_")[-1])

        for attr_tag, attr_cfg in attr_index.items():
            baseline_path = prompt_dir / f"{attr_tag}_baseline.json"
            if baseline_path.exists():
                payload = _load_json(baseline_path)
                rows.append(
                    {
                        "prompt_idx": prompt_idx,
                        "attr_tag": attr_tag,
                        "attribution": attr_cfg["name"],
                        "variant_tag": "baseline",
                        "method_family": "baseline",
                        "n_components": np.nan,
                        "variant_label": "Baseline",
                        "soft_ns_mean": float(payload.get("soft_ns_mean", np.nan)),
                        "soft_nc_mean": float(payload.get("soft_nc_mean", np.nan)),
                    }
                )

            for dimred_tag, dimred_cfg in dimred_index.items():
                json_path = prompt_dir / f"{attr_tag}_dimred_{dimred_tag}.json"
                if not json_path.exists():
                    continue

                payload = _load_json(json_path)
                n_components = dimred_cfg.get("params", {}).get("n_components", np.nan)
                rows.append(
                    {
                        "prompt_idx": prompt_idx,
                        "attr_tag": attr_tag,
                        "attribution": attr_cfg["name"],
                        "variant_tag": dimred_tag,
                        "method_family": dimred_cfg["name"],
                        "n_components": n_components,
                        "variant_label": f"{_pretty_name(dimred_cfg['name'])}\n(k={n_components})",
                        "soft_ns_mean": float(payload.get("soft_ns_mean", np.nan)),
                        "soft_nc_mean": float(payload.get("soft_nc_mean", np.nan)),
                    }
                )

    return pd.DataFrame(rows), attr_index, dimred_index


def _aggregate_variants(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["attr_tag", "attribution", "variant_tag", "method_family", "n_components", "variant_label"],
            dropna=False,
            as_index=False,
        )
        .agg(
            soft_ns_mean=("soft_ns_mean", "mean"),
            soft_ns_se=("soft_ns_mean", _sem),
            soft_nc_mean=("soft_nc_mean", "mean"),
            soft_nc_se=("soft_nc_mean", _sem),
            n_prompts=("prompt_idx", "nunique"),
        )
    )
    return summary


def _variant_order(dimred_index: dict[str, Any]) -> list[str]:
    return ["baseline", *list(dimred_index.keys())]


def _family_order(dimred_index: dict[str, Any]) -> list[str]:
    ordered = ["baseline"]
    for cfg in dimred_index.values():
        family = cfg["name"]
        if family not in ordered:
            ordered.append(family)
    return ordered


def _plot_metric_per_attribution(
    summary_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    attr_index: dict[str, Any],
    dimred_index: dict[str, Any],
    metric_col: str,
    error_col: str,
) -> None:
    variant_order = _variant_order(dimred_index)
    metric_label = "Soft NS" if metric_col == "soft_ns_mean" else "Soft NC"

    for attr_tag, attr_cfg in attr_index.items():
        attr_df = summary_df[summary_df["attr_tag"] == attr_tag].copy()
        if attr_df.empty:
            continue

        attr_df["variant_order"] = attr_df["variant_tag"].map({tag: i for i, tag in enumerate(variant_order)})
        attr_df = attr_df.sort_values("variant_order")

        x = np.arange(len(attr_df))
        values = attr_df[metric_col].to_numpy(dtype=float)
        errors = attr_df[error_col].to_numpy(dtype=float)
        labels = attr_df["variant_label"].tolist()
        counts = attr_df["n_prompts"].tolist()

        colors = ["#5C677D" if tag == "baseline" else "#0B6E4F" for tag in attr_df["variant_tag"]]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.85), 5.8), constrained_layout=True)
        bars = ax.bar(x, values, yerr=errors, capsize=4, color=colors, edgecolor="#1F2933", linewidth=0.8)

        for bar, count, value, err in zip(bars, counts, values, errors):
            y = 0.0 if np.isnan(value) else value + err + 0.01
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{metric_label} by dim-red variant | {_pretty_name(attr_cfg['name'])} | {run_label}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(bottom=0.0)

        output_path = out_dir / f"{run_label}__{attr_tag}__{metric_col}_bar.png"
        fig.savefig(output_path, dpi=220)
        plt.close(fig)


def _pick_best_variant_per_family(summary_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    best_rows: list[pd.Series] = []
    for (_, _), group in summary_df.groupby(["attr_tag", "method_family"], dropna=False):
        ordered = group.sort_values(
            by=[metric_col, "n_components"],
            ascending=[False, True],
            na_position="last",
        )
        best_rows.append(ordered.iloc[0])
    return pd.DataFrame(best_rows).reset_index(drop=True)


def _plot_heatmap_best_per_family(
    best_df: pd.DataFrame,
    out_dir: Path,
    run_label: str,
    attr_index: dict[str, Any],
    dimred_index: dict[str, Any],
    metric_col: str,
) -> None:
    family_order = _family_order(dimred_index)
    attr_order = list(attr_index.keys())
    metric_label = "Soft NS" if metric_col == "soft_ns_mean" else "Soft NC"

    mat = np.full((len(attr_order), len(family_order)), np.nan, dtype=float)
    annotations: list[list[str]] = [["" for _ in family_order] for _ in attr_order]

    family_to_x = {family: i for i, family in enumerate(family_order)}
    attr_to_y = {attr: i for i, attr in enumerate(attr_order)}

    for _, row in best_df.iterrows():
        attr_tag = row["attr_tag"]
        family = row["method_family"]
        if attr_tag not in attr_to_y or family not in family_to_x:
            continue

        y = attr_to_y[attr_tag]
        x = family_to_x[family]
        value = float(row[metric_col])
        mat[y, x] = value

        if family == "baseline" or pd.isna(row["n_components"]):
            annotations[y][x] = f"{value:.3f}"
        else:
            annotations[y][x] = f"{value:.3f}\nk={int(row['n_components'])}"

    cmap = plt.cm.YlGn
    fig, ax = plt.subplots(
        figsize=(max(8, len(family_order) * 1.35), max(4.5, len(attr_order) * 0.8)),
        constrained_layout=True,
    )
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=np.nanmax(mat) if np.isfinite(mat).any() else 1.0)

    ax.set_xticks(np.arange(len(family_order)))
    ax.set_xticklabels([_pretty_name(family) for family in family_order], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(attr_order)))
    ax.set_yticklabels([_pretty_name(attr_index[attr]["name"]) for attr in attr_order])
    ax.set_title(f"{metric_label} heatmap | best n_components per dim-red family | {run_label}")

    for y in range(len(attr_order)):
        for x in range(len(family_order)):
            if np.isnan(mat[y, x]):
                ax.text(x, y, "NA", ha="center", va="center", color="#334E68", fontsize=9)
            else:
                ax.text(x, y, annotations[y][x], ha="center", va="center", color="#102A43", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label)

    output_path = out_dir / f"{run_label}__{metric_col}_heatmap_best_per_family.png"
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create dim-red comparison plots for one model/dataset run directory.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory like outputs/qwen_qwen3-0.6b/tellmewhy",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Output directory for plots. Default: <run-dir>/plots_dimred_summary",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    plot_dir = Path(args.plot_dir) if args.plot_dir else run_dir / "plots_dimred_summary"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df, attr_index, dimred_index = _collect_records(run_dir)
    if df.empty:
        raise ValueError(f"No result JSON files found under: {run_dir}")

    run_label = _run_label(run_dir)
    summary_df = _aggregate_variants(df)
    summary_df.to_csv(plot_dir / f"{run_label}__summary_by_variant.csv", index=False)

    for metric_col, error_col in [("soft_ns_mean", "soft_ns_se"), ("soft_nc_mean", "soft_nc_se")]:
        _plot_metric_per_attribution(
            summary_df=summary_df,
            out_dir=plot_dir,
            run_label=run_label,
            attr_index=attr_index,
            dimred_index=dimred_index,
            metric_col=metric_col,
            error_col=error_col,
        )
        best_df = _pick_best_variant_per_family(summary_df, metric_col)
        best_df.to_csv(plot_dir / f"{run_label}__{metric_col}_best_per_family.csv", index=False)
        _plot_heatmap_best_per_family(
            best_df=best_df,
            out_dir=plot_dir,
            run_label=run_label,
            attr_index=attr_index,
            dimred_index=dimred_index,
            metric_col=metric_col,
        )

    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
