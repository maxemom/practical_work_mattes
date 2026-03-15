from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pwm.utils_pipeline import safe_name


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_prompt_records(run_dir: Path) -> pd.DataFrame:
    attr_index = _load_json(run_dir / "attr_index.json")
    dimred_index = _load_json(run_dir / "dimred_index.json")
    prompt_dirs = sorted((run_dir / "prompts").glob("prompt_*"))
    rows: list[dict[str, Any]] = []

    for prompt_dir in prompt_dirs:
        prompt_name = prompt_dir.name
        prompt_idx = int(prompt_name.split("_")[-1])
        for a_tag, a_cfg in attr_index.items():
            baseline_path = prompt_dir / f"{a_tag}_baseline.json"
            if baseline_path.exists():
                res = _load_json(baseline_path)
                rows.append(
                    {
                        "prompt_idx": prompt_idx,
                        "attr_tag": a_tag,
                        "attribution": a_cfg["name"],
                        "aggregation_tag": "baseline",
                        "aggregation": "baseline",
                        "soft_ns_mean": float(res.get("soft_ns_mean", np.nan)),
                        "soft_nc_mean": float(res.get("soft_nc_mean", np.nan)),
                    }
                )

            for d_tag, d_cfg in dimred_index.items():
                dim_path = prompt_dir / f"{a_tag}_dimred_{d_tag}.json"
                if not dim_path.exists():
                    continue
                res = _load_json(dim_path)
                rows.append(
                    {
                        "prompt_idx": prompt_idx,
                        "attr_tag": a_tag,
                        "attribution": a_cfg["name"],
                        "aggregation_tag": d_tag,
                        "aggregation": d_cfg["name"],
                        "soft_ns_mean": float(res.get("soft_ns_mean", np.nan)),
                        "soft_nc_mean": float(res.get("soft_nc_mean", np.nan)),
                    }
                )
    return pd.DataFrame(rows)


def _plot_6_2_aggregate(df: pd.DataFrame, out_dir: Path, run_label: str) -> None:
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = (
        df.groupby(["attribution", "aggregation"], as_index=False)[["soft_ns_mean", "soft_nc_mean"]]
        .mean()
        .sort_values(["attribution", "aggregation"])
    )
    grouped.to_csv(out_dir / f"{run_label}_section_6_2_summary.csv", index=False)

    attrs = list(grouped["attribution"].drop_duplicates())
    aggs = list(grouped["aggregation"].drop_duplicates())
    x = np.arange(len(attrs))
    width = 0.8 / max(1, len(aggs))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    for j, metric in enumerate(["soft_nc_mean", "soft_ns_mean"]):
        ax = axes[j]
        for i, agg in enumerate(aggs):
            vals = []
            for attr in attrs:
                hit = grouped[(grouped["attribution"] == attr) & (grouped["aggregation"] == agg)]
                vals.append(float(hit[metric].iloc[0]) if not hit.empty else np.nan)
            ax.bar(x + (i - (len(aggs) - 1) / 2) * width, vals, width=width, label=agg)
        ax.set_xticks(x)
        ax.set_xticklabels(attrs, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by attribution/aggregation")
        ax.grid(axis="y", alpha=0.2)

    axes[1].legend(title="aggregation", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.suptitle(f"Section 6.2-style aggregate results | {run_label}")
    fig.savefig(out_dir / f"{run_label}_section_6_2_aggregate.png", dpi=180)
    plt.close(fig)


def _plot_6_3_visual_examples(
    run_dir: Path,
    out_dir: Path,
    run_label: str,
    prompt_idx: int,
    attr_name: str | None,
    metric_col: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    attr_index = _load_json(run_dir / "attr_index.json")
    dimred_index = _load_json(run_dir / "dimred_index.json")

    selected_attr_tag = None
    for a_tag, a_cfg in attr_index.items():
        if attr_name is None or a_cfg["name"].lower() == attr_name.lower():
            selected_attr_tag = a_tag
            break
    if selected_attr_tag is None:
        return

    prompt_dir = run_dir / "prompts" / f"prompt_{prompt_idx:03d}"
    if not prompt_dir.exists():
        return

    curves: list[tuple[str, pd.DataFrame]] = []
    base_csv = prompt_dir / f"{selected_attr_tag}_baseline_steps.csv"
    if base_csv.exists():
        curves.append(("baseline", pd.read_csv(base_csv)))

    for d_tag, d_cfg in dimred_index.items():
        p = prompt_dir / f"{selected_attr_tag}_dimred_{d_tag}_steps.csv"
        if p.exists():
            curves.append((d_cfg["name"], pd.read_csv(p)))

    curves = [(name, frame) for name, frame in curves if metric_col in frame.columns]
    if not curves:
        return

    # Line chart over generation steps.
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for name, frame in curves:
        ax.plot(frame["target_pos"].values, frame[metric_col].values, marker="o", linewidth=1.4, label=name)
    ax.set_title(f"Section 6.3-style step curves | {run_label} | prompt_{prompt_idx:03d} | {selected_attr_tag}")
    ax.set_xlabel("target_pos")
    ax.set_ylabel(metric_col)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(out_dir / f"{run_label}_section_6_3_prompt_{prompt_idx:03d}_{selected_attr_tag}_{metric_col}_lines.png", dpi=180)
    plt.close(fig)

    # Heatmap over (aggregation x generation step).
    names = [name for name, _ in curves]
    step_axis = curves[0][1]["target_pos"].tolist()
    mat = []
    for _, frame in curves:
        idx = frame.set_index("target_pos")
        row = [float(idx.loc[s, metric_col]) if s in idx.index else np.nan for s in step_axis]
        mat.append(row)
    mat_np = np.array(mat, dtype=float)

    fig2, ax2 = plt.subplots(figsize=(max(8, len(step_axis) * 0.6), max(3, len(names) * 0.45)), constrained_layout=True)
    im = ax2.imshow(mat_np, aspect="auto", cmap="viridis")
    ax2.set_yticks(np.arange(len(names)))
    ax2.set_yticklabels(names)
    ax2.set_xticks(np.arange(len(step_axis)))
    ax2.set_xticklabels(step_axis, rotation=45, ha="right")
    ax2.set_xlabel("target_pos")
    ax2.set_title(f"{metric_col} heatmap")
    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label(metric_col)
    fig2.savefig(
        out_dir / f"{run_label}_section_6_3_prompt_{prompt_idx:03d}_{selected_attr_tag}_{metric_col}_heatmap.png",
        dpi=180,
    )
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="configs/base.yaml")
    parser.add_argument("--grid", type=str, default="configs/grid.yaml")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--plot-dir", type=str, default=None)
    parser.add_argument("--prompt-idx", type=int, default=0)
    parser.add_argument("--attr-name", type=str, default=None, help="Optional attribution method for section 6.3 plot.")
    parser.add_argument("--metric-col", type=str, default="soft_nc", choices=["soft_nc", "soft_ns", "dP0", "dPR", "dPnotR"])
    args = parser.parse_args()

    base_cfg = _load_yaml(Path(args.base))
    grid_cfg = _load_yaml(Path(args.grid))
    output_root = Path(base_cfg["paths"]["output_dir"])
    plot_root = Path(args.plot_dir) if args.plot_dir else output_root / "plots" / "section_6_2_6_3"
    plot_root.mkdir(parents=True, exist_ok=True)

    models = grid_cfg.get("models", [])
    datasets = grid_cfg.get("datasets", [])

    for m in models:
        for d in datasets:
            model_name = m.get("name", "")
            dataset_name = d.get("name", "")
            if args.model_name and model_name != args.model_name:
                continue
            if args.dataset_name and dataset_name != args.dataset_name:
                continue

            run_dir = output_root / safe_name(model_name) / safe_name(dataset_name)
            if not run_dir.exists():
                print(f"[skip] missing run dir: {run_dir}")
                continue

            run_label = f"{safe_name(model_name)}__{safe_name(dataset_name)}"
            run_plot_dir = plot_root / run_label
            run_plot_dir.mkdir(parents=True, exist_ok=True)

            df = _collect_prompt_records(run_dir)
            if df.empty:
                print(f"[skip] no metric files found in: {run_dir}")
                continue

            df.to_csv(run_plot_dir / "all_prompt_level_records.csv", index=False)
            _plot_6_2_aggregate(df=df, out_dir=run_plot_dir, run_label=run_label)
            _plot_6_3_visual_examples(
                run_dir=run_dir,
                out_dir=run_plot_dir,
                run_label=run_label,
                prompt_idx=args.prompt_idx,
                attr_name=args.attr_name,
                metric_col=args.metric_col,
            )
            print(f"[ok] wrote plots to {run_plot_dir}")


if __name__ == "__main__":
    main()
