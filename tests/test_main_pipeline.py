from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

pytest.importorskip("matplotlib")
pytest.importorskip("numpy")
pytest.importorskip("pandas")

import main as pipeline_main


def test_run_pipeline_uses_create_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []

    monkeypatch.setattr(
        pipeline_main.prepare_dataset,
        "run_all_from_config",
        lambda base_path: calls.append(("prepare", {"base_path": base_path})),
    )
    monkeypatch.setattr(
        pipeline_main.run_grid,
        "run_grid",
        lambda **kwargs: calls.append(("grid", dict(kwargs))),
    )
    monkeypatch.setattr(
        pipeline_main.create_plots,
        "create_all_plots",
        lambda **kwargs: calls.append(("plots", dict(kwargs))),
    )

    pipeline_main.run_pipeline(
        base_path="configs/base.yaml",
        grid_path="configs/grid.yaml",
        max_prompts=3,
        skip_prepare=False,
    )

    assert calls[0] == ("prepare", {"base_path": "configs/base.yaml"})
    assert calls[1] == (
        "grid",
        {
            "base_path": "configs/base.yaml",
            "grid_path": "configs/grid.yaml",
            "max_prompts": 3,
        },
    )
    assert calls[2][0] == "plots"
    assert calls[2][1]["output_root"] == "outputs"
    assert calls[2][1]["grid_path"] == "configs/grid.yaml"
    assert calls[2][1]["all_bar_dimreds"] is True
    assert calls[2][1]["all_token_attributions"] is True
