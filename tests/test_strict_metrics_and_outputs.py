from __future__ import annotations

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

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from pwm.typess import MethodResult
from pwm.utils_dimred import reduce_raw_targets_to_importance
from pwm.utils_metrics_strict import compute_strict_soft_metrics
from scripts.interpret_outputs import build_long_form
from scripts.run_grid import build_combo_aggregates, save_prompt_outputs, summarize_aggregate
from pwm.typess import PromptRunResult


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 13, hidden_size: int = 6) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        h = x.cumsum(dim=1)
        logits = self.proj(h)
        return type("ForwardOutput", (), {"logits": logits})()


def test_reduce_raw_targets_to_importance_normalizes_columns() -> None:
    raw_targets = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 1.0]],
            [[0.0, 2.0], [2.0, 0.0]],
            [[float("nan"), float("nan")], [0.0, 3.0]],
        ],
        dtype=torch.float32,
    )

    importance = reduce_raw_targets_to_importance(raw_targets, "baseline", {"norm": "l2"}, seed=42)

    assert torch.allclose(torch.nansum(importance[:, 0]), torch.tensor(1.0))
    assert torch.allclose(torch.nansum(importance[:, 1]), torch.tensor(1.0))
    assert math.isnan(float(importance[2, 0].item()))


def test_compute_strict_soft_metrics_is_reproducible() -> None:
    torch.manual_seed(0)
    model = TinyCausalLM()
    total_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    importance = torch.tensor(
        [
            [0.6, 0.4],
            [0.4, 0.3],
            [float("nan"), 0.3],
            [float("nan"), float("nan")],
        ],
        dtype=torch.float32,
    )

    result_a = compute_strict_soft_metrics(model, total_ids, 2, importance, seed=77)
    result_b = compute_strict_soft_metrics(model, total_ids, 2, importance, seed=77)

    assert result_a.soft_ns_per_token == pytest.approx(result_b.soft_ns_per_token)
    assert result_a.soft_nc_per_token == pytest.approx(result_b.soft_nc_per_token)
    assert result_a.final_sufficiency_per_token == pytest.approx(result_b.final_sufficiency_per_token)
    assert result_a.final_comprehensiveness_per_token == pytest.approx(result_b.final_comprehensiveness_per_token)


def test_summarize_aggregate_uses_prompt_level_means() -> None:
    aggregates = build_combo_aggregates(
        model_name="demo-model",
        dataset_name="demo-dataset",
        run_meta={"device": "cpu"},
        attrs=[{"tag": "saliency", "name": "saliency", "params": {}}],
        dimreds=[{"tag": "baseline", "name": "baseline", "params": {"norm": "l2"}}],
    )

    aggregate = aggregates["saliency__baseline"]
    aggregate.prompts = [
        type(
            "PromptRecord",
            (),
            {
                "method_result": MethodResult(
                    combo_key="saliency__baseline",
                    attribution_tag="saliency",
                    attribution_name="saliency",
                    attribution_params={},
                    dimred_tag="baseline",
                    dimred_name="baseline",
                    dimred_params={"norm": "l2"},
                    importance_scores=[],
                    soft_ns_per_token=[0.1],
                    soft_nc_per_token=[0.2],
                    final_sufficiency_per_token=[0.3],
                    final_comprehensiveness_per_token=[0.4],
                    random_soft_ns_per_token=[0.05],
                    random_soft_nc_per_token=[0.1],
                    soft_ns_mean=0.1,
                    soft_nc_mean=0.2,
                    final_sufficiency_mean=0.3,
                    final_comprehensiveness_mean=0.4,
                )
            },
        )(),
        type(
            "PromptRecord",
            (),
            {
                "method_result": MethodResult(
                    combo_key="saliency__baseline",
                    attribution_tag="saliency",
                    attribution_name="saliency",
                    attribution_params={},
                    dimred_tag="baseline",
                    dimred_name="baseline",
                    dimred_params={"norm": "l2"},
                    importance_scores=[],
                    soft_ns_per_token=[],
                    soft_nc_per_token=[],
                    final_sufficiency_per_token=[],
                    final_comprehensiveness_per_token=[],
                    random_soft_ns_per_token=[],
                    random_soft_nc_per_token=[],
                    soft_ns_mean=0.0,
                    soft_nc_mean=0.0,
                    final_sufficiency_mean=0.0,
                    final_comprehensiveness_mean=0.0,
                    skipped=True,
                    skip_reason="demo_skip",
                )
            },
        )(),
    ]

    summary = summarize_aggregate(aggregate)

    assert summary["prompt_count"] == 2
    assert summary["successful_prompt_count"] == 1
    assert summary["skipped_prompt_count"] == 1
    assert summary["final_sufficiency_mean"] == pytest.approx(0.3)
    assert summary["skip_reasons"] == ["demo_skip"]


def test_save_prompt_outputs_writes_promptwise_layout(tmp_path: Path) -> None:
    prompt_result = PromptRunResult(
        prompt_idx=0,
        prompt="demo prompt",
        model_name="demo-model",
        dataset_name="demo-dataset",
        generated_text="demo prompt answer",
        source_ids=[1, 2],
        total_ids=[1, 2, 3],
        generated_token_ids=[3],
        source_len=2,
        total_len=3,
        generated_tokens=[" answer"],
        combinations={
            "saliency__baseline": MethodResult(
                combo_key="saliency__baseline",
                attribution_tag="saliency",
                attribution_name="saliency",
                attribution_params={},
                dimred_tag="baseline",
                dimred_name="baseline",
                dimred_params={"norm": "l2"},
                importance_scores=[[0.7], [0.3], [None]],
                soft_ns_per_token=[0.2],
                soft_nc_per_token=[0.4],
                final_sufficiency_per_token=[0.1],
                final_comprehensiveness_per_token=[0.3],
                random_soft_ns_per_token=[0.15],
                random_soft_nc_per_token=[0.35],
                soft_ns_mean=0.2,
                soft_nc_mean=0.4,
                final_sufficiency_mean=0.1,
                final_comprehensiveness_mean=0.3,
                target_pos=[2],
                target_token_ids=[3],
                target_token_texts=[" answer"],
            )
        },
    )

    save_prompt_outputs(tmp_path, prompt_result)

    prompt_dir = tmp_path / "prompts" / "prompt_000"
    assert (prompt_dir / "prompt.json").exists()
    assert (prompt_dir / "saliency_baseline.json").exists()


def test_build_long_form_reads_promptwise_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "demo_model" / "demo_dataset"
    prompt_dir = run_dir / "prompts" / "prompt_000"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    (prompt_dir / "prompt.json").write_text(
        """
{
  "prompt_idx": 0,
  "prompt": "demo prompt",
  "model_name": "demo-model",
  "dataset_name": "demo-dataset",
  "generated_text": "demo prompt answer",
  "source_ids": [1, 2],
  "total_ids": [1, 2, 3],
  "generated_token_ids": [3],
  "generated_tokens": [" answer"],
  "source_len": 2,
  "total_len": 3,
  "warnings": []
}
        """.strip(),
        encoding="utf-8",
    )
    (prompt_dir / "saliency_baseline.json").write_text(
        """
{
  "combo_key": "saliency__baseline",
  "attribution_name": "saliency",
  "dimred_name": "baseline",
  "soft_ns_mean": 0.2,
  "soft_nc_mean": 0.4,
  "final_sufficiency_mean": 0.1,
  "final_comprehensiveness_mean": 0.3,
  "skipped": false,
  "skip_reason": null
}
        """.strip(),
        encoding="utf-8",
    )

    df = build_long_form(tmp_path)

    assert df.shape[0] == 1
    row = df.iloc[0]
    assert row["model_name"] == "demo-model"
    assert row["dataset_name"] == "demo-dataset"
    assert row["attribution"] == "saliency"
    assert row["dimred"] == "baseline"
    assert int(row["prompt_idx"]) == 0
