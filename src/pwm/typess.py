# Data structures
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class Run:
    model: Dict[str, Any]
    dataset: Dict[str, Any]
    attribution: Dict[str, Any]
    dimred: Dict[str, Any]
    run_id: str
    resolved: Dict[str, Any]