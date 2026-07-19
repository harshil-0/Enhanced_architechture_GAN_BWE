import random
import numpy as np
import torch

def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set random seed across Python, NumPy, PyTorch CPU, and CUDA for reproducible runs.

    Args:
        seed: Random seed value.
        deterministic: If True, sets CuDNN to deterministic mode.

    Returns:
        The seed integer value that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[MLOps] Reproducible random seed set to: {seed}")
    return seed
