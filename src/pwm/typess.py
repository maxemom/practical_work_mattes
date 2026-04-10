from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Run:
    model: Dict[str, Any]
    dataset: Dict[str, Any]
    attribution: Dict[str, Any]
    dimred: Dict[str, Any]
    run_id: str
    resolved: Dict[str, Any]


@dataclass
class MethodResult:
    combo_key: str
    attribution_tag: str
    attribution_name: str
    attribution_params: Dict[str, Any]
    dimred_tag: str
    dimred_name: str
    dimred_params: Dict[str, Any]
    importance_scores: List[List[Optional[float]]]
    soft_ns_per_token: List[float]
    soft_nc_per_token: List[float]
    final_sufficiency_per_token: List[float]
    final_comprehensiveness_per_token: List[float]
    random_soft_ns_per_token: List[float]
    random_soft_nc_per_token: List[float]
    soft_ns_mean: float
    soft_nc_mean: float
    final_sufficiency_mean: float
    final_comprehensiveness_mean: float
    target_pos: List[int] = field(default_factory=list)
    target_token_ids: List[int] = field(default_factory=list)
    target_token_texts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    attribution_device: Optional[str] = None
    attribution_elapsed_ms: Optional[float] = None
    attribution_step_times_ms: List[float] = field(default_factory=list)
    dimred_elapsed_ms: Optional[float] = None
    metrics_elapsed_ms: Optional[float] = None
    combo_elapsed_ms: Optional[float] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass
class PromptRunResult:
    prompt_idx: int
    prompt: str
    model_name: str
    dataset_name: str
    generated_text: str
    source_ids: List[int]
    total_ids: List[int]
    generated_token_ids: List[int]
    source_len: int
    total_len: int
    generated_tokens: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    combinations: Dict[str, MethodResult] = field(default_factory=dict)


@dataclass
class PromptCombinationRecord:
    prompt_idx: int
    prompt: str
    generated_text: str
    source_ids: List[int]
    total_ids: List[int]
    generated_token_ids: List[int]
    source_len: int
    total_len: int
    method_result: MethodResult
    generated_tokens: List[str] = field(default_factory=list)


@dataclass
class ComboAggregateResult:
    combo_key: str
    model_name: str
    dataset_name: str
    attribution_tag: str
    attribution_name: str
    attribution_params: Dict[str, Any]
    dimred_tag: str
    dimred_name: str
    dimred_params: Dict[str, Any]
    run_meta: Dict[str, Any]
    prompts: List[PromptCombinationRecord] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
