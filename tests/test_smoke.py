from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

torch = pytest.importorskip("torch")

from pwm.utils_pipeline import hf_generate_once


class _DummyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token = "<eos>"
    pad_token_id = 1

    def __call__(self, prompt: str, return_tensors: str = "pt", padding: bool = False, truncation: bool = False):
        del prompt, return_tensors, padding, truncation
        return {
            "input_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens: bool = False):
        del skip_special_tokens
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(str(int(tok)) for tok in ids)


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        pad_token_id: int,
    ) -> torch.Tensor:
        del attention_mask, max_new_tokens, do_sample, temperature, top_p, pad_token_id
        extra = torch.tensor([[5, 6]], dtype=torch.long, device=input_ids.device)
        return torch.cat([input_ids, extra], dim=1)


def test_hf_generate_once_returns_regular_tensors() -> None:
    model = _DummyModel()
    tokenizer = _DummyTokenizer()

    source_ids, generated_ids, full_text = hf_generate_once(
        model=model,
        tokenizer=tokenizer,
        prompt="demo",
        generation_cfg={"max_new_tokens": 2, "do_sample": False, "temperature": 1.0, "top_p": 1.0},
    )

    assert source_ids.tolist() == [2, 3, 4]
    assert generated_ids.tolist() == [2, 3, 4, 5, 6]
    assert full_text == "2 3 4 5 6"
    assert not source_ids.is_inference()
    assert not generated_ids.is_inference()
