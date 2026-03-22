from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from pwm.utils_dimred_V2 import reduce_raw_target
from pwm.utils_pipeline import build_attr_index, build_dimred_index
from pwm.utils_results_V2 import save_baseline_result_v2, save_dimred_result_v2
from pwm.utils_runtime import build_resolved_run_config


def _base_cfg() -> dict:
    return {
        "paths": {"output_dir": "outputs"},
        "seeds": {"seed": 42},
        "runtime": {"device": "auto"},
        "generation": {
            "max_new_tokens": 10,
            "temperature": 1.0,
            "do_sample": False,
            "top_p": 1.0,
        },
    }


def _make_tensor() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 3, 5, dtype=torch.float32)


def test_build_resolved_run_config_keeps_base_generation_defaults() -> None:
    resolved = build_resolved_run_config(
        base_cfg=_base_cfg(),
        model_cfg={"name": "demo", "params": {"max_new_tokens": 2, "dtype": "bfloat16"}},
        dataset_cfg={"name": "tiny", "path": "data.txt"},
        attrs=[{"name": "saliency", "params": {}}],
        dimreds=[{"name": "pca", "params": {"n_components": 1}}],
        chosen_device="cpu",
    )

    assert resolved["generation"] == {
        "max_new_tokens": 2,
        "temperature": 1.0,
        "do_sample": False,
        "top_p": 1.0,
    }
    assert resolved["runtime"]["device"] == "cpu"
    assert resolved["model"]["params"]["dtype"] == "bfloat16"
    assert "dtype" not in resolved["generation"]


def test_build_resolved_run_config_applies_temperature_override() -> None:
    resolved = build_resolved_run_config(
        base_cfg=_base_cfg(),
        model_cfg={"name": "demo", "params": {"temperature": 0.7}},
        dataset_cfg={"name": "tiny", "path": "data.txt"},
        attrs=[{"name": "saliency", "params": {}}],
        dimreds=[{"name": "pca", "params": {"n_components": 1}}],
        chosen_device="cpu",
    )

    assert resolved["generation"]["temperature"] == 0.7
    assert resolved["generation"]["max_new_tokens"] == 10


def test_reduce_raw_target_ica_is_reproducible_with_seed() -> None:
    x = _make_tensor()

    out_a = reduce_raw_target(x, "ica", {"n_components": 2}, seed=42)
    out_b = reduce_raw_target(x, "ica", {"n_components": 2}, seed=42)

    assert torch.allclose(out_a, out_b)


def test_reduce_raw_target_respects_explicit_random_state() -> None:
    x = _make_tensor()
    params = {"n_components": 2, "random_state": 7}

    out_a = reduce_raw_target(x, "ica", params, seed=42)
    out_b = reduce_raw_target(x, "ica", params, seed=99)

    assert torch.allclose(out_a, out_b)


def test_reduce_raw_target_accepts_grid_method_names() -> None:
    x = _make_tensor()

    factor_analysis = reduce_raw_target(x, "factor_analysis", {"n_components": 1}, seed=42)
    kernel_pca = reduce_raw_target(x, "kernel_pca", {"n_components": 1}, seed=42)

    assert factor_analysis.shape == (4, 3)
    assert kernel_pca.shape == (4, 3)


def test_build_indices_use_readable_tags() -> None:
    attrs = build_attr_index(
        [
            {"name": "saliency", "params": {}},
            {"name": "saliency", "params": {}},
            {"name": "input_x_gradient", "params": {}},
        ]
    )
    dimreds = build_dimred_index(
        [
            {"name": "pca", "params": {"n_components": 1}},
            {"name": "pca", "params": {"n_components": 2}},
            {"name": "nmf", "params": {}},
        ]
    )

    assert list(attrs.keys()) == ["saliency", "saliency_2", "input_x_gradient"]
    assert list(dimreds.keys()) == ["pca_n_components_1", "pca_n_components_2", "nmf"]


def test_v2_result_writers_only_create_json(tmp_path: Path) -> None:
    result = {
        "soft_ns_mean": 0.5,
        "soft_nc_mean": 0.25,
        "target_pos": [1],
        "target_token_id": [7],
    }

    save_baseline_result_v2(tmp_path, "saliency", result)
    save_dimred_result_v2(tmp_path, "saliency", "pca_n_components_2", result)

    assert (tmp_path / "saliency_baseline.json").exists()
    assert (tmp_path / "saliency_dimred_pca_n_components_2.json").exists()
    assert not list(tmp_path.glob("*.csv"))
