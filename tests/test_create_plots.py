from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")
pytest.importorskip("pandas")

from scripts.create_plots import (
    _available_attr_tags,
    _available_dimred_tags,
    _selected_prompt_indices,
    compute_baseline_comparison_stats,
    create_all_plots,
    filter_to_n_components_per_dimred,
    filter_to_best_n_components_per_dimred,
    load_run_records,
    plot_baseline_bars,
    plot_baseline_stat_heatmaps,
    plot_dimred_bars_per_attribution,
    plot_importance_distribution_heatmaps,
    plot_metric_heatmaps,
    plot_metric_heatmaps_all_dimreds,
    plot_n_components_comparison,
    plot_top1_share_soft_comp_correlation,
    plot_token_attribution_rows,
    set_paper_plot_style,
    summarize_importance_distribution,
    summarize_records,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_run_dir(
    tmp_path: Path,
    *,
    model_slug: str = "demo_model",
    dataset_slug: str = "demo_dataset",
    model_name: str = "demo-model",
    dataset_name: str = "demo-dataset",
) -> Path:
    run_dir = tmp_path / model_slug / dataset_slug
    _write_json(
        run_dir / "run_meta.json",
        {"model_name": model_name, "dataset_name": dataset_name},
    )
    _write_json(
        run_dir / "attr_index.json",
        {
            "saliency": {"name": "saliency", "params": {}, "index": 0},
            "deeplift": {"name": "deeplift", "params": {}, "index": 1},
        },
    )
    _write_json(
        run_dir / "dimred_index.json",
        {
            "baseline": {"name": "baseline", "params": {"norm": "l2"}, "index": 0},
            "pca_n_components_1": {"name": "pca", "params": {"n_components": 1}, "index": 1},
            "pca_n_components_3": {"name": "pca", "params": {"n_components": 3}, "index": 2},
        },
    )

    prompt_payloads = {
        0: {
            "prompt_idx": 0,
            "prompt": "demo prompt zero",
            "model_name": model_name,
            "dataset_name": dataset_name,
            "generated_text": "demo prompt zero answer",
            "source_ids": [10, 11],
            "total_ids": [10, 11, 12, 13],
            "generated_token_ids": [12, 13],
            "generated_tokens": [" ans", " wer"],
            "source_len": 2,
            "total_len": 4,
            "warnings": [],
        },
        1: {
            "prompt_idx": 1,
            "prompt": "demo prompt one",
            "model_name": model_name,
            "dataset_name": dataset_name,
            "generated_text": "demo prompt one answer",
            "source_ids": [20, 21],
            "total_ids": [20, 21, 22, 23],
            "generated_token_ids": [22, 23],
            "generated_tokens": [" foo", " bar"],
            "source_len": 2,
            "total_len": 4,
            "warnings": [],
        },
    }

    method_payloads = {
        0: {
            "saliency_baseline.json": {
                "combo_key": "saliency__baseline",
                "attribution_tag": "saliency",
                "attribution_name": "saliency",
                "attribution_params": {},
                "dimred_tag": "baseline",
                "dimred_name": "baseline",
                "dimred_params": {"norm": "l2"},
                "importance_scores": [[0.5, 0.2], [0.4, 0.3], [None, 0.1], [None, None]],
                "soft_ns_per_token": [0.1, 0.2],
                "soft_nc_per_token": [0.4, 0.5],
                "final_sufficiency_per_token": [0.2, 0.3],
                "final_comprehensiveness_per_token": [0.6, 0.7],
                "soft_ns_mean": 0.1,
                "soft_nc_mean": 0.4,
                "final_sufficiency_mean": 0.2,
                "final_comprehensiveness_mean": 0.6,
                "target_pos": [2, 3],
                "target_token_ids": [12, 13],
                "target_token_texts": [" ans", " wer"],
                "warnings": [],
                "skipped": False,
                "skip_reason": None,
            },
            "saliency_dimred_pca_n_components_1.json": {
                "combo_key": "saliency__pca_n_components_1",
                "attribution_tag": "saliency",
                "attribution_name": "saliency",
                "attribution_params": {},
                "dimred_tag": "pca_n_components_1",
                "dimred_name": "pca",
                "dimred_params": {"n_components": 1},
                "importance_scores": [[0.3, 0.1], [0.6, 0.2], [None, 0.4], [None, None]],
                "soft_ns_per_token": [0.2, 0.1],
                "soft_nc_per_token": [0.5, 0.5],
                "final_sufficiency_per_token": [0.3, 0.2],
                "final_comprehensiveness_per_token": [0.7, 0.7],
                "soft_ns_mean": 0.2,
                "soft_nc_mean": 0.5,
                "final_sufficiency_mean": 0.3,
                "final_comprehensiveness_mean": 0.7,
                "target_pos": [2, 3],
                "target_token_ids": [12, 13],
                "target_token_texts": [" ans", " wer"],
                "warnings": [],
                "skipped": False,
                "skip_reason": None,
            },
            "saliency_dimred_pca_n_components_3.json": {
                "combo_key": "saliency__pca_n_components_3",
                "attribution_tag": "saliency",
                "attribution_name": "saliency",
                "attribution_params": {},
                "dimred_tag": "pca_n_components_3",
                "dimred_name": "pca",
                "dimred_params": {"n_components": 3},
                "importance_scores": [[0.4, 0.2], [0.5, 0.3], [None, 0.3], [None, None]],
                "soft_ns_per_token": [0.3, 0.2],
                "soft_nc_per_token": [0.6, 0.6],
                "final_sufficiency_per_token": [0.4, 0.3],
                "final_comprehensiveness_per_token": [0.8, 0.8],
                "soft_ns_mean": 0.3,
                "soft_nc_mean": 0.6,
                "final_sufficiency_mean": 0.4,
                "final_comprehensiveness_mean": 0.8,
                "target_pos": [2, 3],
                "target_token_ids": [12, 13],
                "target_token_texts": [" ans", " wer"],
                "warnings": [],
                "skipped": False,
                "skip_reason": None,
            },
            "deeplift_baseline.json": {
                "combo_key": "deeplift__baseline",
                "attribution_tag": "deeplift",
                "attribution_name": "deeplift",
                "attribution_params": {},
                "dimred_tag": "baseline",
                "dimred_name": "baseline",
                "dimred_params": {"norm": "l2"},
                "importance_scores": [[0.2, 0.3], [0.1, 0.5], [None, 0.4], [None, None]],
                "soft_ns_per_token": [0.3, 0.3],
                "soft_nc_per_token": [0.6, 0.6],
                "final_sufficiency_per_token": [0.4, 0.4],
                "final_comprehensiveness_per_token": [0.8, 0.8],
                "soft_ns_mean": 0.3,
                "soft_nc_mean": 0.6,
                "final_sufficiency_mean": 0.4,
                "final_comprehensiveness_mean": 0.8,
                "target_pos": [2, 3],
                "target_token_ids": [12, 13],
                "target_token_texts": [" ans", " wer"],
                "warnings": [],
                "skipped": False,
                "skip_reason": None,
            },
            "deeplift_dimred_pca_n_components_1.json": {
                "combo_key": "deeplift__pca_n_components_1",
                "attribution_tag": "deeplift",
                "attribution_name": "deeplift",
                "attribution_params": {},
                "dimred_tag": "pca_n_components_1",
                "dimred_name": "pca",
                "dimred_params": {"n_components": 1},
                "importance_scores": [],
                "soft_ns_per_token": [],
                "soft_nc_per_token": [],
                "final_sufficiency_per_token": [],
                "final_comprehensiveness_per_token": [],
                "soft_ns_mean": 0.0,
                "soft_nc_mean": 0.0,
                "final_sufficiency_mean": 0.0,
                "final_comprehensiveness_mean": 0.0,
                "target_pos": [],
                "target_token_ids": [],
                "target_token_texts": [],
                "warnings": [],
                "skipped": True,
                "skip_reason": "demo_skip",
            },
        },
        1: {
            "saliency_baseline.json": {
                "combo_key": "saliency__baseline",
                "attribution_tag": "saliency",
                "attribution_name": "saliency",
                "attribution_params": {},
                "dimred_tag": "baseline",
                "dimred_name": "baseline",
                "dimred_params": {"norm": "l2"},
                "importance_scores": [[0.6, 0.3], [0.2, 0.4], [None, 0.2], [None, None]],
                "soft_ns_per_token": [0.5, 0.5],
                "soft_nc_per_token": [0.8, 0.8],
                "final_sufficiency_per_token": [0.5, 0.5],
                "final_comprehensiveness_per_token": [0.9, 0.9],
                "soft_ns_mean": 0.5,
                "soft_nc_mean": 0.8,
                "final_sufficiency_mean": 0.5,
                "final_comprehensiveness_mean": 0.9,
                "target_pos": [2, 3],
                "target_token_ids": [22, 23],
                "target_token_texts": [" foo", " bar"],
                "warnings": [],
                "skipped": False,
                "skip_reason": None,
            },
            "saliency_dimred_pca_n_components_1.json": {
                "combo_key": "saliency__pca_n_components_1",
                "attribution_tag": "saliency",
                "attribution_name": "saliency",
                "attribution_params": {},
                "dimred_tag": "pca_n_components_1",
                "dimred_name": "pca",
                "dimred_params": {"n_components": 1},
                "importance_scores": [[0.4, 0.5], [0.3, 0.6], [None, 0.3], [None, None]],
                "soft_ns_per_token": [0.4, 0.4],
                "soft_nc_per_token": [0.7, 0.7],
                "final_sufficiency_per_token": [0.4, 0.4],
                "final_comprehensiveness_per_token": [0.8, 0.8],
                "soft_ns_mean": 0.4,
                "soft_nc_mean": 0.7,
                "final_sufficiency_mean": 0.4,
                "final_comprehensiveness_mean": 0.8,
                "target_pos": [2, 3],
                "target_token_ids": [22, 23],
                "target_token_texts": [" foo", " bar"],
                "warnings": [],
                "skipped": False,
                "skip_reason": None,
            },
            "saliency_dimred_pca_n_components_3.json": {
                "combo_key": "saliency__pca_n_components_3",
                "attribution_tag": "saliency",
                "attribution_name": "saliency",
                "attribution_params": {},
                "dimred_tag": "pca_n_components_3",
                "dimred_name": "pca",
                "dimred_params": {"n_components": 3},
                "importance_scores": [[0.5, 0.6], [0.4, 0.7], [None, 0.4], [None, None]],
                "soft_ns_per_token": [0.6, 0.6],
                "soft_nc_per_token": [0.9, 0.9],
                "final_sufficiency_per_token": [0.6, 0.6],
                "final_comprehensiveness_per_token": [1.0, 1.0],
                "soft_ns_mean": 0.6,
                "soft_nc_mean": 0.9,
                "final_sufficiency_mean": 0.6,
                "final_comprehensiveness_mean": 1.0,
                "target_pos": [2, 3],
                "target_token_ids": [22, 23],
                "target_token_texts": [" foo", " bar"],
                "warnings": [],
                "skipped": False,
                "skip_reason": None,
            },
            "deeplift_baseline.json": {
                "combo_key": "deeplift__baseline",
                "attribution_tag": "deeplift",
                "attribution_name": "deeplift",
                "attribution_params": {},
                "dimred_tag": "baseline",
                "dimred_name": "baseline",
                "dimred_params": {"norm": "l2"},
                "importance_scores": [[0.1, 0.2], [0.4, 0.4], [None, 0.5], [None, None]],
                "soft_ns_per_token": [0.2, 0.2],
                "soft_nc_per_token": [0.3, 0.3],
                "final_sufficiency_per_token": [0.2, 0.2],
                "final_comprehensiveness_per_token": [0.4, 0.4],
                "soft_ns_mean": 0.2,
                "soft_nc_mean": 0.3,
                "final_sufficiency_mean": 0.2,
                "final_comprehensiveness_mean": 0.4,
                "target_pos": [2, 3],
                "target_token_ids": [22, 23],
                "target_token_texts": [" foo", " bar"],
                "warnings": [],
                "skipped": False,
                "skip_reason": None,
            },
            "deeplift_dimred_pca_n_components_1.json": {
                "combo_key": "deeplift__pca_n_components_1",
                "attribution_tag": "deeplift",
                "attribution_name": "deeplift",
                "attribution_params": {},
                "dimred_tag": "pca_n_components_1",
                "dimred_name": "pca",
                "dimred_params": {"n_components": 1},
                "importance_scores": [[0.3, 0.2], [0.5, 0.4], [None, 0.2], [None, None]],
                "soft_ns_per_token": [0.1, 0.1],
                "soft_nc_per_token": [0.2, 0.2],
                "final_sufficiency_per_token": [0.1, 0.1],
                "final_comprehensiveness_per_token": [0.2, 0.2],
                "soft_ns_mean": 0.1,
                "soft_nc_mean": 0.2,
                "final_sufficiency_mean": 0.1,
                "final_comprehensiveness_mean": 0.2,
                "target_pos": [2, 3],
                "target_token_ids": [22, 23],
                "target_token_texts": [" foo", " bar"],
                "warnings": [],
                "skipped": False,
                "skip_reason": None,
            },
        },
    }

    for prompt_idx, prompt_payload in prompt_payloads.items():
        prompt_dir = run_dir / "prompts" / f"prompt_{prompt_idx:03d}"
        _write_json(prompt_dir / "prompt.json", prompt_payload)
        for file_name, payload in method_payloads[prompt_idx].items():
            _write_json(prompt_dir / file_name, payload)

    return run_dir


def test_selected_prompt_indices_supports_all_single_and_range() -> None:
    available = [0, 1, 2]

    assert _selected_prompt_indices(available) == [0, 1, 2]
    assert _selected_prompt_indices(available, prompt_idx=1) == [1]
    assert _selected_prompt_indices(available, prompt_range=(1, 2)) == [1, 2]

    with pytest.raises(ValueError):
        _selected_prompt_indices(available, prompt_idx=4)


def test_summarize_records_aggregates_selected_prompts(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    run = load_run_records(run_dir)

    assert run.records.shape[0] == 10

    all_summary = summarize_records(run.records)
    saliency_baseline = all_summary[
        (all_summary["attribution_tag"] == "saliency") & (all_summary["dimred_tag"] == "baseline")
    ].iloc[0]
    assert saliency_baseline["prompt_count"] == 2
    assert saliency_baseline["soft_suff_mean"] == pytest.approx(0.3)
    assert saliency_baseline["soft_comp_mean"] == pytest.approx(0.6)
    assert saliency_baseline["soft_suff_std"] == pytest.approx(math.sqrt(0.08), rel=1e-6)

    prompt_one_summary = summarize_records(run.records[run.records["prompt_idx"] == 1])
    deeplift_pca = prompt_one_summary[
        (prompt_one_summary["attribution_tag"] == "deeplift")
        & (prompt_one_summary["dimred_tag"] == "pca_n_components_1")
    ].iloc[0]
    assert deeplift_pca["prompt_count"] == 1
    assert deeplift_pca["soft_suff_mean"] == pytest.approx(0.1)
    assert deeplift_pca["soft_suff_std"] == pytest.approx(0.0)


def test_available_attr_and_dimred_tags_cover_present_records(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    run = load_run_records(run_dir)
    summary = summarize_records(run.records)

    assert _available_attr_tags(run, run.records) == ["saliency", "deeplift"]
    assert _available_dimred_tags(run, summary) == ["baseline", "pca_n_components_1", "pca_n_components_3"]


def test_filter_to_best_n_components_keeps_baseline_and_best_component(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    run = load_run_records(run_dir)

    filtered = filter_to_best_n_components_per_dimred(run, run.records)
    dimred_tags = filtered["dimred_tag"].dropna().astype(str).drop_duplicates().tolist()

    assert dimred_tags == ["baseline", "pca_n_components_3"]


def test_filter_to_n_components_keeps_baseline_and_requested_component(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    run = load_run_records(run_dir)

    filtered = filter_to_n_components_per_dimred(run, run.records, n_components=1)
    dimred_tags = filtered["dimred_tag"].dropna().astype(str).drop_duplicates().tolist()

    assert dimred_tags == ["baseline", "pca_n_components_1"]
    assert _available_dimred_tags(run, filtered) == ["baseline", "pca_n_components_1"]


def test_compute_baseline_comparison_stats_pairs_against_l2_baseline(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    run = load_run_records(run_dir)

    stats = compute_baseline_comparison_stats(run.records)

    saliency_pca = stats[
        (stats["attribution_tag"] == "saliency")
        & (stats["dimred_tag"] == "pca_n_components_1")
    ].iloc[0]
    baseline_row = stats[
        (stats["attribution_tag"] == "saliency")
        & (stats["dimred_tag"] == "baseline")
    ].iloc[0]

    assert saliency_pca["paired_prompt_count"] == 2
    assert saliency_pca["soft_suff_delta_mean"] == pytest.approx(0.0)
    assert saliency_pca["soft_comp_delta_mean"] == pytest.approx(0.0)
    assert saliency_pca["soft_suff_nonzero_pairs"] == 2
    assert saliency_pca["soft_suff_test"] == "paired_sign_test"
    assert baseline_row["soft_suff_delta_mean"] == pytest.approx(0.0)
    assert baseline_row["soft_comp_delta_mean"] == pytest.approx(0.0)


def test_summarize_importance_distribution_captures_spread_across_k(tmp_path: Path) -> None:
    run_dir = _build_run_dir(tmp_path)
    run = load_run_records(run_dir)

    distribution = summarize_importance_distribution(run.records)

    saliency_baseline = distribution[
        (distribution["attribution_tag"] == "saliency")
        & (distribution["dimred_tag"] == "baseline")
    ].iloc[0]
    saliency_pca_3 = distribution[
        (distribution["attribution_tag"] == "saliency")
        & (distribution["dimred_tag"] == "pca_n_components_3")
    ].iloc[0]

    assert saliency_baseline["distribution_sample_count"] == 4
    assert saliency_baseline["top3_share_mean"] == pytest.approx(1.0)
    assert saliency_pca_3["top1_share_mean"] < saliency_baseline["top1_share_mean"]
    assert saliency_pca_3["normalized_entropy_mean"] > saliency_baseline["normalized_entropy_mean"]


def test_create_all_plots_respects_grid_model_dataset_selection(tmp_path: Path) -> None:
    set_paper_plot_style()
    _build_run_dir(
        tmp_path / "outputs",
        model_slug="demo-model",
        dataset_slug="demo-dataset",
        model_name="demo-model",
        dataset_name="demo-dataset",
    )
    _build_run_dir(
        tmp_path / "outputs",
        model_slug="other-model",
        dataset_slug="other-dataset",
        model_name="other-model",
        dataset_name="other-dataset",
    )

    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        """
models:
  - name: demo-model
datasets:
  - name: demo-dataset
""".strip(),
        encoding="utf-8",
    )

    report = create_all_plots(
        output_root=str(tmp_path / "outputs"),
        plot_dir=str(tmp_path / "plots"),
        grid_path=str(grid_path),
        all_bar_dimreds=True,
        all_token_attributions=True,
        local_files_only=True,
    )

    assert report["run_count"] == 1
    assert report["runs"][0]["run_label"] == "demo-model__demo-dataset"
    assert (tmp_path / "plots" / "demo-model__demo-dataset").exists()
    assert not (tmp_path / "plots" / "other-model__other-dataset").exists()


def test_create_plots_writes_all_plot_types(tmp_path: Path) -> None:
    set_paper_plot_style()
    run_dir = _build_run_dir(tmp_path)
    run = load_run_records(run_dir)
    selected_prompt_indices = _selected_prompt_indices(run.records["prompt_idx"].tolist())
    summary = summarize_records(run.records[run.records["prompt_idx"].isin(selected_prompt_indices)])
    stats = compute_baseline_comparison_stats(run.records[run.records["prompt_idx"].isin(selected_prompt_indices)])
    plot_dir = tmp_path / "plots"

    bars_path = plot_baseline_bars(run, summary, selected_prompt_indices, plot_dir)
    stat_path = plot_baseline_stat_heatmaps(run, stats, selected_prompt_indices, plot_dir)
    heatmap_path = plot_metric_heatmaps(run, summary, selected_prompt_indices, plot_dir)
    full_heatmap_path = plot_metric_heatmaps_all_dimreds(run, summary, selected_prompt_indices, plot_dir)
    distribution = summarize_importance_distribution(run.records[run.records["prompt_idx"].isin(selected_prompt_indices)])
    importance_distribution_paths = plot_importance_distribution_heatmaps(run, distribution, selected_prompt_indices, plot_dir)
    correlation_path = plot_top1_share_soft_comp_correlation(run, distribution, summary, selected_prompt_indices, plot_dir)
    component_path = plot_n_components_comparison(run, run.records, selected_prompt_indices, plot_dir)
    dimred_bar_paths = plot_dimred_bars_per_attribution(run, summary, selected_prompt_indices, plot_dir)
    token_path = plot_token_attribution_rows(
        run,
        run.records,
        selected_prompt_indices,
        plot_dir,
        token_attr_query="saliency",
        token_dimred_queries=["baseline", "pca_n_components_1"],
        local_files_only=True,
    )

    assert bars_path is not None and bars_path.exists()
    assert stat_path is not None and stat_path.exists()
    assert heatmap_path is not None and heatmap_path.exists()
    assert full_heatmap_path is not None and full_heatmap_path.exists()
    assert len(importance_distribution_paths) == 3
    assert all(path.exists() for path in importance_distribution_paths)
    assert correlation_path is not None and correlation_path.exists()
    assert component_path is not None and component_path.exists()
    assert len(dimred_bar_paths) == 2
    assert all(path.exists() for path in dimred_bar_paths)
    assert token_path is not None and token_path.exists()


def test_create_all_plots_writes_special_attribution_plots_only_for_special_dataset(tmp_path: Path) -> None:
    set_paper_plot_style()
    _build_run_dir(
        tmp_path / "outputs",
        model_slug="demo-model",
        dataset_slug="special-dataset",
        model_name="demo-model",
        dataset_name="Sentences_for_attribution_plots",
    )
    _build_run_dir(
        tmp_path / "outputs",
        model_slug="demo-model",
        dataset_slug="regular-dataset",
        model_name="demo-model",
        dataset_name="demo-dataset",
    )

    report = create_all_plots(
        output_root=str(tmp_path / "outputs"),
        plot_dir=str(tmp_path / "plots"),
        all_bar_dimreds=False,
        all_token_attributions=False,
        local_files_only=True,
    )

    runs = {entry["run_label"]: entry for entry in report["runs"]}
    special_entry = runs["demo-model__sentences_for_attribution_plots"]
    regular_entry = runs["demo-model__demo-dataset"]

    special_paths = [Path(path) for path in special_entry["generated_paths"]["special_attribution"]]
    assert len(special_paths) == 9
    assert all(path.exists() for path in special_paths)
    assert all("Special Attribution" in str(path) for path in special_paths)
    assert all("__target_2__" in path.name for path in special_paths)
    assert {path.parent.name for path in special_paths} == {"prompt_000", "prompt_001"}
    assert any(path.name.startswith("saliency__baseline__") for path in special_paths)
    assert len(special_entry["generated_paths"]["dimred_bars_per_attribution"]) == 2
    assert len(special_entry["generated_paths"]["heatmaps_all_dimreds"]) == 1
    assert len(special_entry["generated_paths"]["importance_spread"]) == 3
    assert len(special_entry["generated_paths"]["soft_comp_correlation"]) == 1

    assert regular_entry["generated_paths"]["special_attribution"] == []
