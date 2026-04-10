from __future__ import annotations

import importlib.util
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
from pwm.utils_pipeline import (
    build_attr_index,
    build_dimred_index,
    patch_lxt_attention_interface_compatibility,
    patch_lxt_transformers_compatibility,
    register_inseq_model_configs,
)
from pwm.utils_results_V2 import save_baseline_result_v2, save_dimred_result_v2
from pwm.utils_runtime import build_resolved_run_config, resolve_model_dtype


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


def test_resolve_model_dtype_prefers_explicit_model_dtype() -> None:
    resolved = build_resolved_run_config(
        base_cfg=_base_cfg(),
        model_cfg={"name": "demo", "params": {"dtype": "bfloat16"}},
        dataset_cfg={"name": "tiny", "path": "data.txt"},
        attrs=[{"name": "saliency", "params": {}}],
        dimreds=[{"name": "pca", "params": {"n_components": 1}}],
        chosen_device="cpu",
    )

    dtype, dtype_name = resolve_model_dtype(resolved)

    assert dtype == torch.bfloat16
    assert dtype_name == "bfloat16"


def test_resolve_model_dtype_defaults_to_fp32_on_mps() -> None:
    resolved = build_resolved_run_config(
        base_cfg=_base_cfg(),
        model_cfg={"name": "demo", "params": {}},
        dataset_cfg={"name": "tiny", "path": "data.txt"},
        attrs=[{"name": "saliency", "params": {}}],
        dimreds=[{"name": "pca", "params": {"n_components": 1}}],
        chosen_device="mps",
    )

    dtype, dtype_name = resolve_model_dtype(resolved)

    assert dtype == torch.float32
    assert dtype_name == "float32"


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


def test_patch_lxt_transformers_compatibility_restores_removed_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("transformers")
    import transformers.pytorch_utils as pytorch_utils
    from transformers.models.roberta import modeling_roberta

    monkeypatch.delattr(pytorch_utils, "find_pruneable_heads_and_indices", raising=False)
    monkeypatch.delattr(modeling_roberta, "RobertaSdpaSelfAttention", raising=False)

    patch_lxt_transformers_compatibility()

    assert hasattr(pytorch_utils, "find_pruneable_heads_and_indices")
    heads, index = pytorch_utils.find_pruneable_heads_and_indices({1}, 4, 2, set())
    assert heads == {1}
    assert index.tolist() == [0, 1, 4, 5, 6, 7]
    assert modeling_roberta.RobertaSdpaSelfAttention is modeling_roberta.RobertaSelfAttention


def test_patch_lxt_transformers_compatibility_allows_lxt_import() -> None:
    pytest.importorskip("transformers")
    if importlib.util.find_spec("lxt") is None:
        pytest.skip("lxt is not installed in this environment")

    patch_lxt_transformers_compatibility()
    from lxt.efficient import monkey_patch

    assert callable(monkey_patch)


def test_patch_lxt_attention_interface_compatibility_preserves_attention_interface() -> None:
    pytest.importorskip("transformers")
    if importlib.util.find_spec("lxt") is None:
        pytest.skip("lxt is not installed in this environment")

    patch_lxt_transformers_compatibility()
    patch_lxt_attention_interface_compatibility()

    from lxt.efficient import patches as lxt_patches
    from lxt.efficient.models import qwen2 as lxt_qwen2
    from transformers.models.qwen2 import modeling_qwen2

    assert lxt_qwen2.attnLRP[modeling_qwen2] is lxt_patches.patch_attention
    assert lxt_qwen2.cp_LRP[modeling_qwen2] is lxt_patches.patch_cp_attention

    def eager_attention_forward(module, query, key, value, *args, **kwargs):
        del module, query, key, value, args, kwargs
        return "ok"

    class FakeAttentionInterface(dict):
        def get_interface(self, attn_implementation, default):
            del attn_implementation
            return default

    fake_module = type(
        "FakeAttentionModule",
        (),
        {
            "eager_attention_forward": staticmethod(eager_attention_forward),
            "ALL_ATTENTION_FUNCTIONS": FakeAttentionInterface({"sdpa": eager_attention_forward}),
        },
    )()

    original_interface = fake_module.ALL_ATTENTION_FUNCTIONS
    success = lxt_patches.patch_attention(fake_module)

    assert success is True
    assert fake_module.ALL_ATTENTION_FUNCTIONS is original_interface
    assert hasattr(fake_module.ALL_ATTENTION_FUNCTIONS, "get_interface")
    assert callable(fake_module.ALL_ATTENTION_FUNCTIONS["sdpa"])


def test_register_inseq_model_configs_patches_value_zeroing_tensor_outputs() -> None:
    pytest.importorskip("inseq")
    register_inseq_model_configs()
    from inseq.attr.feat.ops.value_zeroing import ValueZeroing

    clean_state = torch.ones(2, 3, dtype=torch.float32)
    dummy = type(
        "DummyValueZeroing",
        (),
        {
            "clean_block_output_states": {0: clean_state},
            "corrupted_block_output_states": {},
        },
    )()

    hook = ValueZeroing.get_states_extract_and_patch_hook(dummy, 0, 0)
    output = torch.randn(2, 3, dtype=torch.float32)
    returned = hook(None, None, output)

    assert torch.allclose(dummy.corrupted_block_output_states[0], output.float())
    assert torch.allclose(returned, clean_state)

    tuple_output = (torch.randn(2, 3, dtype=torch.float32), "tail")
    returned_tuple = hook(None, None, tuple_output)

    assert isinstance(returned_tuple, tuple)
    assert torch.allclose(returned_tuple[0], clean_state)
    assert returned_tuple[1] == "tail"
