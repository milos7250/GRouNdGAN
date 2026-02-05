import os
import random
from secrets import randbits

import numpy as np

from loggers import setup_logger

random_seed = randbits(32)
logger = setup_logger(__name__)
logger.info(f"Initial random seed: {random_seed}")


def set_seeds(seed: int | None = None) -> None:
    """
    Set the random seed for Python's random and NumPy to ensure
    deterministic behavior.
    """
    global random_seed
    if seed is not None:
        random_seed = seed

    random.seed(random_seed)
    np.random.seed(random_seed)
    logger.info(f"Deterministic mode enabled, using seed {random_seed}.")


def set_pytorch_seeds(seed: int | None = None, deterministic: bool = False) -> None:
    """
    Set PyTorch random seed. Optionally enable deterministic algorithms for reproducibility.
    """
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    global random_seed
    if seed is not None:
        random_seed = seed

    import torch

    torch.manual_seed(random_seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info(f"PyTorch deterministic mode enabled, using seed {random_seed}.")
