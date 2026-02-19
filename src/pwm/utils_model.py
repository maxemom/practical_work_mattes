from ast import Dict
from typing import Any, Dict, List
import inseq
from inseq.models import attribution_model
from torch import device
from torch.cuda import temperature
from torch.nn import parameter
import torch

def prepare_inseq(resolved: Dict[str, Any]) -> Any:
    model_name = resolved["model"]["name"]
    attribution_method = resolved["attribution"]["name"]
    device = resolved["runtime"]["device"]

    model = inseq.load_model(
        model_name,
        attribution_method=attribution_method,
        device=device,
    )
    return model

