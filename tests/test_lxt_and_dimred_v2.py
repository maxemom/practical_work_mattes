from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
pytest.importorskip("sklearn")

from pwm.utils_attribution_V2 import get_raw_targets_lxt_v2
from pwm.utils_dimred_V2 import reduce_raw_target
from pwm.main_function import ExperimentRuntime


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 17, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, use_cache=None):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            x = self.embed(input_ids)
        else:
            x = inputs_embeds

        h = x.cumsum(dim=1)
        logits = self.proj(h)
        return type("ForwardOutput", (), {"logits": logits})()


def test_reduce_raw_target_is_step_local() -> None:
    x_a = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, 0.0]],
            [[0.0, 1.0], [0.5, 0.5]],
            [[0.2, 0.8], [0.2, 0.8]],
        ],
        dtype=torch.float32,
    )
    x_b = x_a.clone()
    x_b[:, 1, :] = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 4.0],
            [3.0, 1.0],
            [1.0, 3.0],
        ],
        dtype=torch.float32,
    )

    out_a = reduce_raw_target(x_a, "pca", {"n_components": 1}, seed=42)
    out_b = reduce_raw_target(x_b, "pca", {"n_components": 1}, seed=42)

    assert torch.allclose(out_a[:, 0], out_b[:, 0], atol=1e-6, equal_nan=True)


def test_reduce_raw_target_nmf_uses_absolute_values() -> None:
    x = torch.tensor(
        [
            [[-1.0, 2.0], [3.0, -4.0]],
            [[2.0, -3.0], [-1.0, 2.0]],
            [[-2.0, 1.0], [2.0, -1.0]],
        ],
        dtype=torch.float32,
    )

    out_neg = reduce_raw_target(x, "nmf", {"n_components": 1, "init": "random", "max_iter": 100}, seed=7)
    out_abs = reduce_raw_target(x.abs(), "nmf", {"n_components": 1, "init": "random", "max_iter": 100}, seed=7)

    assert torch.allclose(out_neg, out_abs, atol=1e-6, equal_nan=True)


def test_get_raw_targets_lxt_v2_returns_pipeline_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    efficient_mod = types.ModuleType("lxt.efficient")
    efficient_mod.monkey_patch = lambda model_module, verbose=False: None
    lxt_mod = types.ModuleType("lxt")
    lxt_mod.efficient = efficient_mod
    monkeypatch.setitem(sys.modules, "lxt", lxt_mod)
    monkeypatch.setitem(sys.modules, "lxt.efficient", efficient_mod)

    torch.manual_seed(0)
    model = TinyCausalLM()
    generated_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)

    result = get_raw_targets_lxt_v2(
        model=model,
        generated_ids=generated_ids,
        source_len=3,
    )

    assert result.raw_target.shape == (5, 2, 8)
    assert torch.isnan(result.raw_target[3:, 0, :]).all()
    assert torch.isnan(result.raw_target[4:, 1, :]).all()
    assert result.source_ids_debug.tolist() == [1, 2, 3]
    assert result.target_ids_debug.tolist() == [1, 2, 3, 4, 5]


def test_get_raw_targets_lxt_v2_records_step_times(monkeypatch: pytest.MonkeyPatch) -> None:
    efficient_mod = types.ModuleType("lxt.efficient")
    efficient_mod.monkey_patch = lambda model_module, verbose=False: None
    lxt_mod = types.ModuleType("lxt")
    lxt_mod.efficient = efficient_mod
    monkeypatch.setitem(sys.modules, "lxt", lxt_mod)
    monkeypatch.setitem(sys.modules, "lxt.efficient", efficient_mod)

    torch.manual_seed(0)
    model = TinyCausalLM()
    generated_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)

    result = get_raw_targets_lxt_v2(
        model=model,
        generated_ids=generated_ids,
        source_len=3,
        attr_params={"log_step_times": False},
    )

    assert result.device == "cpu"
    assert result.elapsed_ms >= 0.0
    assert result.step_times_ms is not None
    assert len(result.step_times_ms) == 2
    assert all(step_ms >= 0.0 for step_ms in result.step_times_ms)


class _RecordingWrapper:
    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = device


def test_experiment_runtime_move_to_device_updates_cached_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    hf_model = TinyCausalLM()
    primary = _RecordingWrapper(hf_model, device="cpu")
    cached_model = TinyCausalLM()
    cached = _RecordingWrapper(cached_model, device="cpu")

    runtime = ExperimentRuntime(
        model_name="demo",
        device="cpu",
        model_dtype_name="float32",
        hf_model=hf_model,
        tokenizer=None,
        primary_attr_name="saliency",
        primary_inseq_model=primary,
        can_switch_runtime=False,
        attr_cache={"ig": cached},
    )

    moves: list[str] = []

    def _record_to(self: nn.Module, target_device: str):
        moves.append(target_device)
        return self

    monkeypatch.setattr(TinyCausalLM, "to", _record_to, raising=False)

    runtime.move_to_device("cuda:0")

    assert runtime.device == "cuda:0"
    assert primary.device == "cuda:0"
    assert cached.device == "cuda:0"
    assert moves == ["cuda:0", "cuda:0"]
