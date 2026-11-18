import os
import random
import torch
import numpy as np


def setup_seed(seed=0, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Same seed -> same weights. cuBLAS needs a fixed workspace for deterministic results;
        # this must be set before the first CUDA matmul, so call setup_seed at the very start.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
