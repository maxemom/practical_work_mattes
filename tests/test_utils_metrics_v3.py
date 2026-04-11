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

from pwm.utils_metrics_strict import compute_strict_soft_metrics


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


def test_compute_strict_soft_metrics_exposes_random_baseline_fields() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM()
    source_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    generated_ids = torch.tensor([4, 5], dtype=torch.long)
    total_ids = torch.cat([source_ids, generated_ids])

    result = compute_strict_soft_metrics(
        model,
        total_ids,
        len(source_ids),
        _build_importance_map(),
        seed=123,
    )

    expected_len = len(result.target_pos)
    assert expected_len == len(generated_ids)
    assert len(result.random_soft_ns_per_token) == expected_len
    assert len(result.random_soft_nc_per_token) == expected_len
    assert len(result.final_sufficiency_per_token) == expected_len
    assert len(result.final_comprehensiveness_per_token) == expected_len
    assert math.isfinite(result.soft_ns_mean)
    assert math.isfinite(result.soft_nc_mean)
    assert math.isfinite(result.final_sufficiency_mean)
    assert math.isfinite(result.final_comprehensiveness_mean)


def test_compute_strict_soft_metrics_random_baseline_is_reproducible() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM()
    source_ids = torch.tensor([2, 1, 4], dtype=torch.long)
    generated_ids = torch.tensor([3, 5], dtype=torch.long)
    total_ids = torch.cat([source_ids, generated_ids])
    importance_map = _build_importance_map()

    result_a = compute_strict_soft_metrics(
        model,
        total_ids,
        len(source_ids),
        importance_map,
        seed=77,
    )
    result_b = compute_strict_soft_metrics(
        model,
        total_ids,
        len(source_ids),
        importance_map,
        seed=77,
    )

    assert result_a.random_soft_ns_per_token == pytest.approx(result_b.random_soft_ns_per_token)
    assert result_a.random_soft_nc_per_token == pytest.approx(result_b.random_soft_nc_per_token)
    assert result_a.final_sufficiency_per_token == pytest.approx(result_b.final_sufficiency_per_token)
    assert result_a.final_comprehensiveness_per_token == pytest.approx(result_b.final_comprehensiveness_per_token)
    assert result_a.soft_ns_mean == pytest.approx(result_b.soft_ns_mean)
    assert result_a.soft_nc_mean == pytest.approx(result_b.soft_nc_mean)
    assert result_a.final_sufficiency_mean == pytest.approx(result_b.final_sufficiency_mean)
    assert result_a.final_comprehensiveness_mean == pytest.approx(result_b.final_comprehensiveness_mean)


def test_compute_strict_soft_metrics_log_ratios_are_finite_for_zero_signal() -> None:
    model = ZeroCausalLM()
    source_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    generated_ids = torch.tensor([4, 5], dtype=torch.long)
    total_ids = torch.cat([source_ids, generated_ids])
    importance_map = _build_importance_map()

    result = compute_strict_soft_metrics(
        model,
        total_ids,
        len(source_ids),
        importance_map,
        seed=5,
    )

    assert result.soft_ns_per_token == pytest.approx([0.0, 0.0])
    assert result.soft_nc_per_token == pytest.approx([0.0, 0.0])
    assert result.random_soft_ns_per_token == pytest.approx([0.0, 0.0])
    assert result.random_soft_nc_per_token == pytest.approx([0.0, 0.0])
    assert result.final_sufficiency_per_token == pytest.approx([0.0, 0.0])
    assert result.final_comprehensiveness_per_token == pytest.approx([0.0, 0.0])
    assert math.isfinite(result.final_sufficiency_mean)
    assert math.isfinite(result.final_comprehensiveness_mean)
