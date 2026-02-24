import os
import random
import numpy as np
import torch
from typing import Dict, Any

def set_global_seed(resolved: Dict[str, Any]) -> int:
    seed = int(resolved.get("seeds", {}).get("seed", 42))

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # determinism flags (optional; can slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed