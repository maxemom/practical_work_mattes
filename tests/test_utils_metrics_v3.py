from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from pwm.utils_metrics_V3 import compute_soft_norm_metrics_v3


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 17, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            x = self.embed(input_ids)
        else:
            x = inputs_embeds

        h = x.cumsum(dim=1)
        logits = self.proj(h)
        return type("ForwardOutput", (), {"logits": logits})()


class ZeroCausalLM(TinyCausalLM):
    def __init__(self, vocab_size: int = 11, hidden_size: int = 6) -> None:
        super().__init__(vocab_size=vocab_size, hidden_size=hidden_size)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.proj.weight.zero_()


def _build_importance_map() -> torch.Tensor:
    return torch.tensor(
        [
            [0.30, 0.20],
            [0.10, 0.15],
            [0.60, 0.25],
            [0.00, 0.40],
            [0.00, 0.00],
        ],
        dtype=torch.float32,
    )


def test_compute_soft_norm_metrics_v3_exposes_random_baseline_fields() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM()
    source_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    generated_ids = torch.tensor([4, 5], dtype=torch.long)

    result = compute_soft_norm_metrics_v3(
        model,
        source_ids,
        generated_ids,
        _build_importance_map(),
        seed=123,
        num_mc_samples=4,
        rationale_size_mode="all_mass",
    )

    expected_len = len(result.target_pos)
    assert expected_len == len(generated_ids)
    assert len(result.random_soft_ns) == expected_len
    assert len(result.random_soft_nc) == expected_len
    assert len(result.log_soft_ns_over_random) == expected_len
    assert len(result.log_soft_nc_over_random) == expected_len
    assert math.isfinite(result.random_soft_ns_mean)
    assert math.isfinite(result.random_soft_nc_mean)
    assert math.isfinite(result.log_soft_ns_over_random_mean)
    assert math.isfinite(result.log_soft_nc_over_random_mean)


def test_compute_soft_norm_metrics_v3_random_baseline_is_reproducible() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM()
    source_ids = torch.tensor([2, 1, 4], dtype=torch.long)
    generated_ids = torch.tensor([3, 5], dtype=torch.long)
    importance_map = _build_importance_map()

    result_a = compute_soft_norm_metrics_v3(
        model,
        source_ids,
        generated_ids,
        importance_map,
        seed=77,
        num_mc_samples=5,
        rationale_size_mode="effective_support",
    )
    result_b = compute_soft_norm_metrics_v3(
        model,
        source_ids,
        generated_ids,
        importance_map,
        seed=77,
        num_mc_samples=5,
        rationale_size_mode="effective_support",
    )

    assert result_a.random_soft_ns == pytest.approx(result_b.random_soft_ns)
    assert result_a.random_soft_nc == pytest.approx(result_b.random_soft_nc)
    assert result_a.log_soft_ns_over_random == pytest.approx(result_b.log_soft_ns_over_random)
    assert result_a.log_soft_nc_over_random == pytest.approx(result_b.log_soft_nc_over_random)
    assert result_a.random_soft_ns_mean == pytest.approx(result_b.random_soft_ns_mean)
    assert result_a.random_soft_nc_mean == pytest.approx(result_b.random_soft_nc_mean)
    assert result_a.log_soft_ns_over_random_mean == pytest.approx(result_b.log_soft_ns_over_random_mean)
    assert result_a.log_soft_nc_over_random_mean == pytest.approx(result_b.log_soft_nc_over_random_mean)


def test_compute_soft_norm_metrics_v3_log_ratios_are_finite_for_zero_signal() -> None:
    model = ZeroCausalLM()
    source_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    generated_ids = torch.tensor([4, 5], dtype=torch.long)
    importance_map = _build_importance_map()

    result = compute_soft_norm_metrics_v3(
        model,
        source_ids,
        generated_ids,
        importance_map,
        seed=5,
        num_mc_samples=3,
        rationale_size_mode="fraction",
        rationale_fraction=0.5,
    )

    assert result.soft_ns == pytest.approx([0.0, 0.0])
    assert result.soft_nc == pytest.approx([0.0, 0.0])
    assert result.random_soft_ns == pytest.approx([0.0, 0.0])
    assert result.random_soft_nc == pytest.approx([0.0, 0.0])
    assert result.log_soft_ns_over_random == pytest.approx([0.0, 0.0])
    assert result.log_soft_nc_over_random == pytest.approx([0.0, 0.0])
    assert math.isfinite(result.log_soft_ns_over_random_mean)
    assert math.isfinite(result.log_soft_nc_over_random_mean)
